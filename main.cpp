#include <mpi.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>

using namespace std;


// Neodnorodnost'
double F(double x, double y, double z) {
    return 3 * exp(x + y + z);
}

// Granichnoe uslovie pri x=0
double A0(double y, double z) {
    return exp(y + z);
}

// Granichnoe uslovie pri x=X
double A1(double y, double z, double X) {
    return exp(X + y + z);
}

// Granichnoe uslovie pri y=0
double B0(double x, double z) {
    return exp(x + z);
}

// Granichnoe uslovie pri y=Y
double B1(double x, double z, double Y) {
    return exp(x + Y + z);
}

// Granichnoe uslovie pri z=0
double C0(double x, double y) {
    return exp(x + y);
}

// Granichnoe uslovie pri z=Z
double C1(double x, double y, double Z) {
    return exp(x + y + Z);
}

int main(int argc, char *argv[]) {
    int rank, size;
    double start, end;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int left = rank - 1;
    int right = (rank < size - 1) ? rank + 1 : -1;

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();

    // Vichislenie osnovnih parametrov oblasti reshenija
    int rit = 300, tag = 31;
    double h1 = 0.005, h2 = 0.005, h3 = 0.005;
    double X = 1, Y = 1, Z = 1;
    double w = 1.7;

    int Nx = (int)(X / h1) - 1, Ny = (int)(Y / h2) - 1, Nz = (int)(Z / h3) - 1;

    int r3 = 20, r4 = 20;
    int Q3 = (int)ceil((double)Ny / r3);
    int Q4 = (int)ceil((double)Nz / r4);

    int Q2 = size;
    int r2 = (int)ceil((double)Nx / Q2);

    int igl2 = rank;

    double *U = (double *)calloc((size_t)r2 * r3 * Q3 * r4 * Q4, sizeof(double));
    double *preLeft = (double *)calloc((size_t)r3 * Q3 * r4 * Q4, sizeof(double));
    double *preRight = (double *)calloc((size_t)r3 * Q3 * r4 * Q4, sizeof(double));

    MPI_Status status;

    MPI_Datatype ujk_t;
    MPI_Type_vector(r3, r4, r4 * Q4, MPI_DOUBLE, &ujk_t);
    MPI_Type_commit(&ujk_t);

    MPI_Datatype  prejk_t;
    MPI_Type_vector(r3, r4, r4 * Q4, MPI_DOUBLE, &prejk_t);
    MPI_Type_commit(&prejk_t);

    for (int i1 = 0; i1 < rit; ++i1) {
        if (i1 > 0) {
            if (right != -1) {
//                printf("%d (%d) recvs right from %d\n", rank, i1, right);
                MPI_Recv(preRight, r3 * Q3 * r4 * Q4, MPI_DOUBLE, right, i1, MPI_COMM_WORLD, &status);
//                printf("%d (%d) recved right from %d\n", rank, i1, right);
            }
        }
        for (int igl3 = 0; igl3 < Q3; ++igl3) {
            for (int igl4 = 0; igl4 < Q4; ++igl4) {
                if (left != -1) {
//                printf("%d (%d, %d) recvs left from %d\n", rank, i1, igl4, left);
                    MPI_Recv(preLeft + (igl3 * r3 * Q4 + igl4) * r4, 1, prejk_t, left, (i1 * tag + igl3) * tag + igl4,
                             MPI_COMM_WORLD, &status);
//                printf("%d (%d, %d) recved left from %d\n", rank, i1, igl4, left);
                }

                for (int i2 = 0; i2 < min(r2, Nx - igl2 * r2); ++i2) {
                    for (int i3 = igl3 * r3; i3 < min((igl3 + 1) * r3, Ny); ++i3) {
                        for (int i4 = igl4 * r4; i4 < min((igl4 + 1) * r4, Nz); ++i4) {
                            int i = igl2 * r2 + i2, j = i3, k = i4;
                            double uim, uip, ujm, ujp, ukm, ukp;

                            if (i == 0) {
                                uim = A0((j + 1) * h2, (k + 1) * h3);
                            } else if (i2 == 0) {
                                uim = preLeft[i3 * r4 * Q4 + i4];
                            } else {
                                uim = U[((i2 - 1) * r3 * Q3 + i3) * r4 * Q4 + i4];
                            }

                            if (i == Nx - 1) {
                                uip = A1((j + 1) * h2, (k + 1) * h3, X);
                            } else if (i2 == r2 - 1) {
                                uip = preRight[i3 * r4 * Q4 + i4];
                            } else {
                                uip = U[((i2 + 1) * r3 * Q3 + i3) * r4 * Q4 + i4];
                            }

                            if (j == 0) {
                                ujm = B0((i + 1) * h1, (k + 1) * h3);
                            } else {
                                ujm = U[(i2 * r3 * Q3 + i3 - 1) * r4 * Q4 + i4];
                            }

                            if (j == Ny - 1) {
                                ujp = B1((i + 1) * h1, (k + 1) * h3, Y);
                            } else {
                                ujp = U[(i2 * r3 * Q3 + i3 + 1) * r4 * Q4 + i4];
                            }

                            if (k == 0) {
                                ukm = C0((i + 1) * h1, (j + 1) * h2);
                            } else {
                                ukm = U[(i2 * r3 * Q3 + i3) * r4 * Q4 + i4 - 1];
                            }

                            if (k == Nz - 1) {
                                ukp = C1((i + 1) * h1, (j + 1) * h2, Z);
                            } else {
                                ukp = U[(i2 * r3 * Q3 + i3) * r4 * Q4 + i4 + 1];
                            }

                            double u = U[(i2 * r3 * Q3 + i3) * r4 * Q4 + i4];

                            U[(i2 * r3 * Q3 + i3) * r4 * Q4 + i4] = w * ((uip + uim) / (h1 * h1)
                                                                         + (ujp + ujm) / (h2 * h2)
                                                                         + (ukp + ukm) / (h3 * h3) - F((i + 1) * h1,
                                                                                                       (j + 1) * h2,
                                                                                                       (k + 1) * h3))
                                                                    / (2 / (h1 * h1) + 2 / (h2 * h2) + 2 / (h3 * h3))
                                                                    + (1 - w) * u;
                        }
                    }
                }
                if (right != -1) {
//                printf("%d (%d, %d) sends right to %d\n", rank, i1, igl4, right);
                    MPI_Send(U + (((r2 - 1) * Q3 + igl3) * r3 * Q4 + igl4) * r4, 1, ujk_t, right,
                             (i1 * tag + igl3) * tag + igl4, MPI_COMM_WORLD);
//                printf("%d (%d, %d) sent right to %d\n", rank, i1, igl4, right);
                }
            }
        }
        if (i1 < rit - 1) {
            if (left != -1) {
//                printf("%d (%d) sends left to %d\n", rank, i1, left);
                MPI_Send(U, r3 * Q3 * r4 * Q4, MPI_DOUBLE, left, i1 + 1, MPI_COMM_WORLD);
//                printf("%d (%d) sent left to %d\n", rank, i1, left);
            }
        }
    }

    double *R;
    if (rank == 0) {
        R = (double *)malloc(sizeof(double) * r2 * Q2 * r3 * Q3 * r4 * Q4);
    }

    MPI_Gather(U, r2 * r3 * Q3 * r4 * Q4, MPI_DOUBLE, R, r2 * r3 * Q3 * r4 * Q4, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();

    MPI_Finalize();

    if (rank == 0) {
        printf("Time: %f\n", end - start);

        FILE *f = fopen("output.txt", "w");
        for (int i = 0; i < Nx; ++i) {
            for (int j = 0; j < Ny; ++j) {
                for (int k = 0; k < Nz; ++k) {
                    fprintf(f, "%f ", R[(i * r3 * Q3 + j) * r4 * Q4 + k]);
                }
                fprintf(f, "\n");
            }
            fprintf(f, "-----------------------------------------------\n");
        }
        fclose(f);
    }

    return 0;
}