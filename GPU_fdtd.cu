#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>

#if defined(_OPENMP)
#include <omp.h>
#define GET_TIME() (omp_get_wtime()) // wall time
#else
#define GET_TIME() ((double)clock() / CLOCKS_PER_SEC) // cpu time
#endif

struct data {
    const char *name;
    int nx, ny;
    float dx, dy, *values;
};


#define GET(data, i, j) ((data)->values[(data)->nx * (j) + (i)])
#define SET(data, i, j, val) ((data)->values[(data)->nx * (j) + (i)] = (val))


int init_data(struct data *data, const char *name, int nx, int ny, float dx,
              float dy, float val)
{
    data->name = name;
    data->nx = nx;
    data->ny = ny;
    data->dx = dx;
    data->dy = dy;
    data->values = (float *)malloc(nx * ny * sizeof(float));
    if(!data->values) {
        printf("Error: Could not allocate data\n");
        return 1;
    }
    for(int i = 0; i < nx * ny; i++) data->values[i] = val;
    return 0;
}


void free_data(struct data *data) { free(data->values); }


int write_data_vtk(struct data *data, int step, int rank)
    // VTK (Visualization Toolkit)
    // .vti file output
{
    // ensure name is not too long
    if(strlen(data->name) > 256) {
        printf("Error: data name too long for output VTK file\n");
        return 1;
    }
  
    // create output filename
    char out[512];
    sprintf(out, "simulation_results/%s_rank%d_%d.vti", data->name, rank, step);
    // printf("Ready to save to: %s\n", out); // Ready to save to: simulation_results/ez_rank0_180.vti

    FILE *fp = fopen(out, "wb"); // write or overwrite binary file
    if(!fp) {
        printf("Error: Could not open output VTK file '%s'\n", out);
        return 1;
    }

    uint64_t num_points = data->nx * data->ny; // grid size to save
    uint64_t num_bytes = num_points * sizeof(float); // number of bytes to save

    fprintf(fp, "<?xml version=\"1.0\"?>\n"
                "<VTKFile"
                " type=\"ImageData\""
                " version=\"1.0\""
                " byte_order=\"LittleEndian\""
                " header_type=\"UInt64\""
                ">\n"
                "  <ImageData"
                " WholeExtent=\"0 %d 0 %d 0 %d\""
                " Spacing=\"%lf %lf %lf\""
                " Origin=\"%lf %lf %lf\""
                ">\n"
                "    <Piece Extent=\"0 %d 0 %d 0 %d\">\n"
                "      <PointData Scalars=\"scalar_data\">\n"
                "        <DataArray"
                " type=\"Float32\""
                " Name=\"%s\""
                " format=\"appended\""
                " offset=\"0\""
                ">\n"
                "        </DataArray>\n"
                "      </PointData>\n"
                "    </Piece>\n"
                "  </ImageData>\n"
                "  <AppendedData encoding=\"raw\">\n_",
            data->nx - 1, data->ny - 1, 0,
            data->dx, data->dy, 0.,
            0., 0., 0.,
            data->nx - 1, data->ny - 1, 0,
            data->name);

    fwrite(&num_bytes, sizeof(uint64_t), 1, fp);          // write number of bytes
    fwrite(data->values, sizeof(float), num_points, fp); // write data values

    fprintf(fp, "  </AppendedData>\n"
                "</VTKFile>\n");

    fclose(fp);

  return 0;
}


int write_manifest_vtk(const char *name, float dt, int nt, int sampling_rate,
                       int numranks) // manifest for Paraview, pvd file
{
  char out[512];
  if(strlen(name) > 256) {
    printf("Error: name too long for Paraview manifest file\n");
    return 1;
  }
  sprintf(out, "simulation_results/%s.pvd", name); // groups .vti files

  FILE *fp = fopen(out, "wb");
  if(!fp) {
    printf("Error: Could not open output VTK manifest file '%s'\n", out);
    return 1;
  }

  fprintf(fp, "<VTKFile"
              " type=\"Collection\""
              " version=\"0.1\""
              " byte_order=\"LittleEndian\">\n"
              "  <Collection>\n");

  for(int n = 0; n < nt; n++) {
    if(sampling_rate && !(n % sampling_rate)) {
      float t = n * dt;
      for(int rank = 0; rank < numranks; rank++) {
        fprintf(fp, "    <DataSet"
                    " timestep=\"%g\""
                    " part=\"%d\""
                    " file='%s_rank%d_%d.vti'/>\n",
                t, rank, name, rank, n);
      }
    }
  }

  fprintf(fp, "  </Collection>\n"
              "</VTKFile>\n");
  fclose(fp);
  return 0;
}


__global__ void update_hx_kernel(float *ez, float *hx, int nx, int ny, float chy) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i < nx && j < ny - 1) {
        int hx_idx = i + j * nx;
        int ez_idx1 = i + (j + 1) * nx;
        int ez_idx2 = i + j * nx;
        hx[hx_idx] = hx[hx_idx] - chy * (ez[ez_idx1] - ez[ez_idx2]);
    }
}


__global__ void update_hy_kernel(float *ez, float *hy, int nx, int ny, float chx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i < nx - 1 && j < ny) {
        int hy_idx = i + j * (nx - 1);
        int ez_idx1 = (i + 1) + j * nx;
        int ez_idx2 = i + j * nx;
        hy[hy_idx] = hy[hy_idx] + chx * (ez[ez_idx1] - ez[ez_idx2]);
    }
}


__global__ void update_ez_kernel(float *ez, float *hx, float *hy, int nx, int ny, float cex, float cey, float t, int problem_id)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
        int ez_idx = i + j * nx;
        int hy_idx1 = i + j * (nx - 1);
        int hy_idx2 = (i - 1) + j * (nx - 1);
        int hx_idx1 = i + j * nx;
        int hx_idx2 = i + (j - 1) * nx;

        ez[ez_idx] = ez[ez_idx] +
                     cex * (hy[hy_idx1] - hy[hy_idx2]) -
                     cey * (hx[hx_idx1] - hx[hx_idx2]);
    }

    // if center of the domain, impose source (only one thread concerned)
    if (i == nx / 2 && j == ny / 2) {
        switch(problem_id) {
        case 1:
        case 2:
            // sinusoidal excitation at 2.4 GHz in the middle of the domain
            ez[i + j * nx] = sinf(2. * M_PI * 2.4e9 * t);
            break;
        }
    }
}

int main(int argc, char **argv)
{
    // getting scaling factor argument
    double factor_p = argc > 2 ? sqrt(atof(argv[2])) : 1.0;
    printf("Scaling factor per dimension: %g\n", factor_p);

    int custom_dim = argc > 3 ? atoi(argv[3]) : -1;
    printf("Custom dimension if any: %d\n", custom_dim);

    float dx = 1.f, dy = 1.f, dt = 1.f;
    int nx = 1, ny = 1, nt = 1, sampling_rate = 1;

    float eps = 8.854187817e-12f;
    float mu = 1.2566370614359173e-06f;
    int problem_id = atoi(argv[1]);
    switch(problem_id) {
    case 1: // small size problem
        dx = dy = (3.e8 / 2.4e9) / 20.; // wavelength / 20
        nx = ny = 500;
        dt = 0.5 / (3.e8 * sqrt(1. / (dx * dx) + 1. / (dy * dy))); // cfl / 2
        nt = 500;
        sampling_rate = 5; // save 1 step out of 5
        break;
    case 2: // larger size problem, usable for initial scaling tests
        dx = dy = (3.e8 / 2.4e9) / 40.; // wavelength / 40
        nx = ny = 16000;
        dt = 0.5 / (3.e8 * sqrt(1. / (dx * dx) + 1. / (dy * dy))); // cfl / 2
        nt = 500;
        sampling_rate = 0;
        break;
    default:
        printf("Error: unknown problem id %d\n", problem_id);
        return 1;
    }

    printf("Solving problem %d:\n", problem_id);
    printf(" - space %gm x %gm (dx=%g, dy=%g; nx=%d, ny=%d)\n",
           dx * nx, dy * ny, dx, dy, nx, ny);
    printf(" - time %gs (dt=%g, nt=%d)\n", dt * nt, dt, nt);

    struct data ez, hx, hy;
    if(init_data(&ez, "ez", nx, ny, dx, dy, 0.) ||
       init_data(&hx, "hx", nx, ny - 1, dx, dy, 0.) || 
       init_data(&hy, "hy", nx - 1, ny, dx, dy, 0.)) {
        printf("Error: could not allocate data\n");
        return 1;
    }

    // pushing memory on GPU
    float *d_ez, *d_hx, *d_hy; // device memory pointers
    float bytes_to_mb = 1024. * 1024.;
    size_t size_ez = nx * ny * sizeof(float);
    size_t size_hx = nx * (ny - 1) * sizeof(float);
    size_t size_hy = (nx - 1) * ny * sizeof(float);
    printf("Allocating %.2f MB on GPU...\n", (size_ez + size_hx + size_hy) / (bytes_to_mb));

    cudaMalloc(&d_ez, size_ez); // allocate device memory
    cudaMalloc(&d_hx, size_hx);
    cudaMalloc(&d_hy, size_hy);

    cudaMemcpy(d_ez, ez.values, size_ez, cudaMemcpyHostToDevice); // copy data to device
    cudaMemcpy(d_hx, hx.values, size_hx, cudaMemcpyHostToDevice); // kind (fourth arg) says transfer from cpu to gpu
    cudaMemcpy(d_hy, hy.values, size_hy, cudaMemcpyHostToDevice);

    double start = GET_TIME(); // start timer when computation begins

    // computation loop
    unsigned int threadsInDirection = 16;
    dim3 num_threads(threadsInDirection, threadsInDirection); // 16x16 = 256 threads per block
    dim3 num_blocks((nx + threadsInDirection - 1) / threadsInDirection,
                    (ny + threadsInDirection - 1) / threadsInDirection);
    
    printf("Starting computation with %d blocks and %d threads/block...\n", num_blocks.x * num_blocks.y, num_threads.x * num_threads.y);

    for(int n = 0; n < nt; n++) {
        if(n && (n % (nt / 10)) == 0) {
            double time_sofar = GET_TIME() - start;
            double eta = (nt - n) * time_sofar / n;
            printf("Computing time step %d/%d (ETA: %g seconds)     \r", n, nt, eta);
            fflush(stdout);
        }

        // update hx and hy
        float chy = dt / (dy * mu);
        update_hx_kernel<<<num_blocks, num_threads>>>(d_ez, d_hx, nx, ny, chy);

        float chx = dt / (dx * mu);
        update_hy_kernel<<<num_blocks, num_threads>>>(d_ez, d_hy, nx, ny, chx);

        // update ez
        float cex = dt / (dx * eps), cey = dt / (dy * eps);
        update_ez_kernel<<<num_blocks, num_threads>>>(d_ez, d_hx, d_hy, nx, ny, cex, cey, n * dt, problem_id);
    
        // output step data in VTK format
        if(sampling_rate && !(n % sampling_rate)) {
            cudaMemcpy(ez.values, d_ez, size_ez, cudaMemcpyDeviceToHost);

            if (write_data_vtk(&ez, n, 0) != 0) {
                printf("Error writing VTK data at step %d\n", n);
                // free allocated GPU memory
                cudaFree(d_ez);
                cudaFree(d_hx);
                cudaFree(d_hy);
                return 1;
            }
        }
    }

    // ensure all kernels are finished before stopping the timer
    cudaDeviceSynchronize();
    
    double time = GET_TIME() - start;

    // write VTK manifest, linking to individual step data files
    if (sampling_rate) {
        printf("\nWriting VTK manifest file...\n");
        write_manifest_vtk("ez", dt, nt, sampling_rate, 1);
    }

    printf("\nDone: %g seconds (%g MUpdates/s)\n", time,
           1.e-6 * (float)nx * (float)ny * (float)nt / (float)time);

    free_data(&ez);
    free_data(&hx);
    free_data(&hy);

    // free allocated GPU memory
    cudaFree(d_hx);
    cudaFree(d_hy);
    cudaFree(d_ez);

    return 0;
}