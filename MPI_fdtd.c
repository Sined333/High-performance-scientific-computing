#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <mpi.h>

#if defined(_OPENMP)
#include <omp.h>
#define GET_TIME() (omp_get_wtime()) // wall time
#else
#define GET_TIME() ((double)clock() / CLOCKS_PER_SEC) // cpu time
#endif

struct data {
  const char *name;
  int nx, ny;
  double dx, dy, *values;
};

#define GET(data, i, j) ((data)->values[(data)->nx * (j) + (i)])
#define SET(data, i, j, val) ((data)->values[(data)->nx * (j) + (i)] = (val))

int init_data(struct data *data, const char *name, int nx, int ny, double dx,
              double dy, double val)
{
  data->name = name;
  data->nx = nx;
  data->ny = ny;
  data->dx = dx;
  data->dy = dy;
  data->values = (double *)malloc(nx * ny * sizeof(double));
  if(!data->values) {
    printf("Error: Could not allocate data\n");
    return 1;
  }
  for(int i = 0; i < nx * ny; i++) data->values[i] = val;
  return 0;
}

void free_data(struct data *data) { free(data->values); }

int write_data_vtk(struct data *data, int step, int rank, int nx, int ny, int offset_x, int offset_y, int nx_local, int ny_local)
{
  char out[512];
  if(strlen(data->name) > 256) {
    printf("Error: data name too long for output VTK file\n");
    return 1;
  }
  sprintf(out, "%s_rank%d_%d.vti", data->name, rank, step);

  FILE *fp = fopen(out, "wb");
  if(!fp) {
    printf("Error: Could not open output VTK file '%s'\n", out);
    return 1;
  }

  uint64_t num_points = nx_local * ny_local;
  uint64_t num_bytes = num_points * sizeof(double);

  fprintf(fp, "<?xml version=\"1.0\"?>\n"
              "<VTKFile"
              " type=\"ImageData\""
              " version=\"1.0\""
              " byte_order=\"LittleEndian\""
              " header_type=\"UInt64\""
              ">\n"
              "  <ImageData"
              " WholeExtent=\"%d %d %d %d 0 %d\""
              " Spacing=\"%lf %lf %lf\""
              " Origin=\"%lf %lf %lf\""
              ">\n"
              "    <Piece Extent=\"%d %d %d %d 0 %d\">\n"
              "      <PointData Scalars=\"scalar_data\">\n"
              "        <DataArray"
              " type=\"Float64\""
              " Name=\"%s\""
              " format=\"appended\""
              " offset=\"0\""
              ">\n"
              "        </DataArray>\n"
              "      </PointData>\n"
              "    </Piece>\n"
              "  </ImageData>\n"
              "  <AppendedData encoding=\"raw\">\n_",
          offset_x, offset_x + nx_local - 1, offset_y, offset_y + ny_local - 1, 0,
          data->dx, data->dy, 0.,
          0., 0., 0.,
          offset_x, offset_x + nx_local - 1, offset_y, offset_y + ny_local - 1, 0,
          data->name);

  fwrite(&num_bytes, sizeof(uint64_t), 1, fp);
  fwrite(data->values, sizeof(double), num_points, fp);

  fprintf(fp, "  </AppendedData>\n"
              "</VTKFile>\n");

  fclose(fp);

  return 0;
}

int write_manifest_vtk(const char *name, double dt, int nt, int sampling_rate,
                       int numranks)
{
  char out[512];
  if(strlen(name) > 256) {
    printf("Error: name too long for Paraview manifest file\n");
    return 1;
  }
  sprintf(out, "%s.pvd", name);

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
      double t = n * dt;
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

int main(int argc, char **argv)
{
  if(argc != 2) {
    printf("Usage: %s problem_id\n", argv[0]);
    return 1;
  }

  // MPI Initialization
  // Initialise MPI
  MPI_Init(&argc, &argv);
  // Determine le nombre de rank total
  int num_ranks;
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  // Initialisation de la topologie virtuelle
  int dims[2] = {0, 0};
  int periods[2] = {0, 0};
  MPI_Dims_create(num_ranks, 2, dims);

  MPI_Comm cart_comm;
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);
  // Determine dans quel rank on est et nos coordonnees
  int cart_rank;
  int coord[2];
  MPI_Comm_rank(cart_comm, &cart_rank);
  MPI_Cart_coords(cart_comm, cart_rank, 2, coord);
  // Determine les voisins
  int rank_up, rank_down, rank_left, rank_right;
  MPI_Cart_shift(cart_comm, 1, 1, &rank_down, &rank_up);
  MPI_Cart_shift(cart_comm, 0, 1, &rank_left, &rank_right);

  double dx = 1., dy = 1., dt = 1.;
  int nx = 1, ny = 1, nt = 1, sampling_rate = 1;

  double eps = 8.854187817e-12;
  double mu = 1.2566370614359173e-06;

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
    sampling_rate = 0; // don't save results
    break;
  default:
    printf("Error: unknown problem id %d\n", problem_id);
    MPI_Finalize();
    return 1;
  }

  // Sizes of the little domain
  int nx_local = nx / dims[0];
  int ny_local = ny / dims[1];

  // Offset
  int offset_x = coord[0] * nx_local;
  int offset_y = coord[1] * ny_local;

  if(cart_rank == 0){
    printf("Solving problem %d:\n", problem_id);
    printf(" - space %gm x %gm (dx=%g, dy=%g; nx=%d, ny=%d)\n",
          dx * nx, dy * ny, dx, dy, nx, ny);
    printf(" - time %gs (dt=%g, nt=%d)\n", dt * nt, dt, nt);
  }

  struct data ez, hx, hy;
  if(init_data(&ez, "ez", nx_local, ny_local, dx, dy, 0.) ||
     init_data(&hx, "hx", nx_local, ny_local, dx, dy, 0.) ||
     init_data(&hy, "hy", nx_local, ny_local, dx, dy, 0.)
  ) {
    printf("Error: could not allocate data\n");
    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 1;
  }

  // Allocation of buffer
  // to calcul hx and hy
  double *ez_up = malloc(sizeof(double) * nx_local);
  double *ez_down = malloc(sizeof(double) * nx_local);
  double *ez_right = malloc(sizeof(double) * ny_local);
  double *ez_left = malloc(sizeof(double) * ny_local);
  // to calcul ez
  double *hx_top = malloc(sizeof(double) * nx_local);
  double *hx_bottom = malloc(sizeof(double) * nx_local);
  double *hy_right = malloc(sizeof(double) * ny_local);
  double *hy_left = malloc(sizeof(double) * ny_local);

  double start = GET_TIME();

  MPI_Request recv_requests_e[2];
  MPI_Request send_requests_e[2];

  MPI_Request recv_requests_h[2];
  MPI_Request send_requests_h[2];

  double chy = dt / (dy * mu), chx = dt / (dx * mu);
  double cex = dt / (dx * eps), cey = dt / (dy * eps);

  for(int n = 0; n < nt; n++) {
    if(n && (n % (nt / 10)) == 0) {
      double time_sofar = GET_TIME() - start;
      double eta = (nt - n) * time_sofar / n;
      printf("Computing time step %d/%d (ETA: %g seconds)     \r", n, nt, eta);
      fflush(stdout);
    }
    // Cas de base pour le status pour éviter un warning
    recv_requests_e[0] = MPI_REQUEST_NULL;
    recv_requests_e[1] = MPI_REQUEST_NULL;
    send_requests_e[0] = MPI_REQUEST_NULL;
    send_requests_e[1] = MPI_REQUEST_NULL;

    recv_requests_h[0] = MPI_REQUEST_NULL;
    recv_requests_h[1] = MPI_REQUEST_NULL;
    send_requests_h[0] = MPI_REQUEST_NULL;
    send_requests_h[1] = MPI_REQUEST_NULL;
    
    // MPI Communication for hx and hy
    // hx
    if(rank_down != MPI_PROC_NULL){
        for(int i = 0; i < nx_local; i++)
            ez_down[i] = GET(&ez, i, 0);
        
        MPI_Isend(ez_down, nx_local, MPI_DOUBLE, rank_down, 0, cart_comm, &send_requests_e[0]);
    }
    if(rank_up != MPI_PROC_NULL)
      MPI_Irecv(ez_up,   nx_local, MPI_DOUBLE, rank_up,   0, cart_comm, &recv_requests_e[0]);
    
      // hy
    if(rank_left != MPI_PROC_NULL){
        for(int j = 0; j < ny_local; j++)
            ez_left[j] = GET(&ez, 0, j);

        MPI_Isend(ez_left,  ny_local, MPI_DOUBLE, rank_left,  1, cart_comm, &send_requests_e[1]);
    }
    if(rank_right != MPI_PROC_NULL)
      MPI_Irecv(ez_right, ny_local, MPI_DOUBLE, rank_right, 1, cart_comm, &recv_requests_e[1]);

    // update hx and hy
    for(int j = 0; j < ny_local; j++) {
      for(int i = 0; i < nx_local; i++) {
        double Ez_ij = GET(&ez, i, j);
        if(j < ny_local - 1) {
          double hx_ij =
            GET(&hx, i, j) - chy * (GET(&ez, i, j + 1) - Ez_ij);
          SET(&hx, i, j, hx_ij);
        }
        if(i < nx_local - 1) {
          double hy_ij =
            GET(&hy, i, j) + chx * (GET(&ez, i + 1, j) - Ez_ij);
          SET(&hy, i, j, hy_ij);
        }
      }
    }      

    // wait pour hx et hy request
    MPI_Waitall(2, recv_requests_e, MPI_STATUS_IGNORE);

    // Calcul des hx et hy manquant
    // hx
    if(rank_up != MPI_PROC_NULL){
      int j = ny_local - 1;
      for(int i = 0; i < nx_local; i++){
        double hx_ij =
          GET(&hx, i, j) - chy * (ez_up[i] - GET(&ez, i, j));
        SET(&hx, i, j, hx_ij);
      }
    }
    // hy
    if(rank_right != MPI_PROC_NULL){
      int i = nx_local - 1;
      for(int j = 0; j < ny_local; j++){
        double hy_ij =
          GET(&hy, i, j) + chx * (ez_right[j] - GET(&ez, i, j));
        SET(&hy, i, j, hy_ij);
      }
    }

    MPI_Waitall(2, send_requests_e, MPI_STATUS_IGNORE);

    // ez
    // MPI communication
    if(rank_up != MPI_PROC_NULL){
        for(int i = 0; i < nx_local; i++)
            hx_top[i] = GET(&hx, i, ny_local - 1);
        
        MPI_Isend(hx_top, nx_local, MPI_DOUBLE, rank_up, 2, cart_comm, &send_requests_h[0]);
    }

    if(rank_down != MPI_PROC_NULL)
      MPI_Irecv(hx_bottom,   nx_local, MPI_DOUBLE, rank_down,   2, cart_comm, &recv_requests_h[0]);

    if(rank_right != MPI_PROC_NULL){
        for(int j = 0; j < ny_local; j++)
        hy_right[j] = GET(&hy, nx_local - 1, j);

        MPI_Isend(hy_right,  ny_local, MPI_DOUBLE, rank_right,  3, cart_comm, &send_requests_h[1]);
    }

    if(rank_left != MPI_PROC_NULL)
      MPI_Irecv(hy_left, ny_local, MPI_DOUBLE, rank_left, 3, cart_comm, &recv_requests_h[1]);
    
    // Calcul de ez
    for(int j = 1; j < ny_local; j++) {
      for(int i = 1; i < nx_local; i++) {
        if(i + offset_x < nx - 1 && j + offset_y < ny - 1){
          double ez_ij = GET(&ez, i, j) +
                        cex * (GET(&hy, i, j) - GET(&hy, i - 1, j)) -
                        cey * (GET(&hx, i, j) - GET(&hx, i, j - 1));
          SET(&ez, i, j, ez_ij);
        }
      }
    }

    MPI_Waitall(2, recv_requests_h, MPI_STATUS_IGNORE);

    // Calcul ez manquant
    if(rank_down != MPI_PROC_NULL && rank_left != MPI_PROC_NULL){
      int i = 0, j = 0;
      double ez_ij = GET(&ez, i, j) +
                         cex * (GET(&hy, i, j) - hy_left[0]) -
                         cey * (GET(&hx, i, j) - hx_bottom[0]);
      SET(&ez, i, j, ez_ij);
    }

    if(rank_down != MPI_PROC_NULL){
      int j = 0;
      for(int i = 1; i < nx_local; i++){
        if(offset_x + i < nx - 1){ 
          double ez_ij = GET(&ez, i, j) +
                         cex * (GET(&hy, i, j) - GET(&hy, i - 1, j)) -
                         cey * (GET(&hx, i, j) - hx_bottom[i]);
          SET(&ez, i, j, ez_ij);
        }
      }
    }

    if(rank_left != MPI_PROC_NULL){
      int i = 0; 
      for(int j = 1; j < ny_local; j++){  
        if(offset_y + j < ny - 1){
          double ez_ij = GET(&ez, i, j) +
                         cex * (GET(&hy, i, j) - hy_left[j]) -
                         cey * (GET(&hx, i, j) - GET(&hx, i, j - 1));
          SET(&ez, i, j, ez_ij);
        }
      }
    }

    MPI_Waitall(2, send_requests_h, MPI_STATUS_IGNORE);

    // impose source
    if(offset_x <= nx/2 && nx/2 < offset_x + nx_local && offset_y <= ny/2 && ny/2 < offset_y + ny_local){
        double t = n * dt;
        switch(problem_id) {
        case 1:
        case 2:
        // sinusoidal excitation at 2.4 GHz in the middle of the domain
        SET(&ez, nx/2 - offset_x, ny/2 - offset_y, sin(2. * M_PI * 2.4e9 * t));
        break;
        default:
        printf("Error: unknown source\n");
        break;
        }
    }
    // output step data in VTK format
    if(sampling_rate && !(n % sampling_rate)) {
      write_data_vtk(&ez, n, cart_rank, nx, ny, offset_x, offset_y, nx_local, ny_local);
      //write_data_vtk(&hx, n, cart_rank, nx, ny, offset_x, offset_y, nx_local, ny_local);
      //write_data_vtk(&hy, n, cart_rank, nx, ny, offset_x, offset_y, nx_local, ny_local);
    }
  }

  //write VTK manifest, linking to individual step data files
  if(cart_rank == 0){
    write_manifest_vtk("ez", dt, nt, sampling_rate, num_ranks);
    // write_manifest_vtk("hx", dt, nt, sampling_rate, num_ranks);
    // write_manifest_vtk("hy", dt, nt, sampling_rate, num_ranks);
  
  double time = GET_TIME() - start;
  printf("\nDone: %g seconds (%g MUpdates/s)\n", time,
         1.e-6 * (double)nx * (double)ny * (double)nt / time);
  }

  free_data(&ez);
  free_data(&hx);
  free_data(&hy);
  
  free(ez_up);
  free(ez_down);
  free(ez_right);
  free(ez_left);
  
  free(hx_top);
  free(hx_bottom);
  free(hy_right);
  free(hy_left);

  // Termine l'utilisation de MPI
  MPI_Comm_free(&cart_comm);

  MPI_Finalize();

  return 0;
}
