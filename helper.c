#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <lapacke.h>

// {{{ band_mat routines

struct band_mat{
  long ncol;        /* Number of columns in band matrix            */
  long nbrows;      /* Number of rows (bands in original matrix)   */
  long nbands_up;   /* Number of bands above diagonal           */
  long nbands_low;  /* Number of bands below diagonal           */
  double *array;    /* Storage for the matrix in banded format  */
  /* Internal temporary storage for solving inverse problem */
  long nbrows_inv;  /* Number of rows of inverse matrix   */
  double *array_inv;/* Store the matrix decomposition if this is generated:                */
                    /* this is used to calculate the action of the inverse matrix.         */
                    /* (what is stored is not the inverse matrix but an equivalent object) */
  int *ipiv;        /* Additional inverse information         */
};
typedef struct band_mat band_mat;

/* Initialise a band matrix of a certain size, allocate memory,
   and set the parameters.  */ 
int init_band_mat(band_mat *bmat, long nbands_lower, long nbands_upper, long n_columns) {
  bmat->nbrows = nbands_lower + nbands_upper + 1;
  bmat->ncol   = n_columns;
  bmat->nbands_up = nbands_upper;
  bmat->nbands_low= nbands_lower;
  bmat->array      = (double *) malloc(sizeof(double)*bmat->nbrows*bmat->ncol);
  bmat->nbrows_inv = bmat->nbands_up*2 + bmat->nbands_low + 1;
  bmat->array_inv  = (double *) malloc(sizeof(double)*(bmat->nbrows+bmat->nbands_low)*bmat->ncol);
  bmat->ipiv       = (int *) malloc(sizeof(int)*bmat->ncol);
  if (bmat->array==NULL||bmat->array_inv==NULL) {
    return 0;
  }  
  /* Initialise array to zero */
  long i;
  for (i=0;i<bmat->nbrows*bmat->ncol;i++) {
    bmat->array[i] = 0.0;
  }
  return 1;
};


/* Get a pointer to a location in the band matrix, using
   the row and column indexes of the full matrix.           */
double *getp(band_mat *bmat, long row, long column) {
  int bandno = bmat->nbands_up + row - column;
  if(row<0 || column<0 || row>=bmat->ncol || column>=bmat->ncol ) {
    printf("Indexes out of bounds in getp: %ld %ld %ld \n",row,column,bmat->ncol);
    exit(1);
  }
  return &bmat->array[bmat->nbrows*column + bandno];
}

/* Retrun the value of a location in the band matrix, using
   the row and column indexes of the full matrix.           */
double getv(band_mat *bmat, long row, long column) {
  return *getp(bmat,row,column);
}

double setv(band_mat *bmat, long row, long column, double val) {
  *getp(bmat,row,column) = val;
  return val;
}

/* Solve the equation Ax = b for a matrix a stored in band format
   and x and b real arrays                                          */
int solve_Ax_eq_b(band_mat *bmat, double *x, double *b) {
  /* Copy bmat array into the temporary store */
  int i,bandno;
  for(i=0;i<bmat->ncol;i++) { 
    for (bandno=0;bandno<bmat->nbrows;bandno++) {
      bmat->array_inv[bmat->nbrows_inv*i+(bandno+bmat->nbands_low)] = bmat->array[bmat->nbrows*i+bandno];
    }
    x[i] = b[i];
  }

  long nrhs = 1;
  long ldab = bmat->nbands_low*2 + bmat->nbands_up + 1;
  int info = LAPACKE_dgbsv( LAPACK_COL_MAJOR, bmat->ncol, bmat->nbands_low, bmat->nbands_up, nrhs, bmat->array_inv, ldab, bmat->ipiv, x, bmat->ncol);
  return info;
}

int printmat(band_mat *bmat) {
  long i,j;
  for(i=0; i<bmat->ncol;i++) {
    for(j=0; j<bmat->nbrows; j++) {
       printf("%ld %ld %g \n",i,j,bmat->array[bmat->nbrows*i + j]);
    }
  }
  return 0;
}
//  }}}

void read_arrays(double *k, double *S, long N);
void read_params(double *L, long *N, double *v, double *t);

int main() {
    // Parameters
    double L;
    long N;
    double v, t;
    read_params(&L, &N, &v, &t);
    printf("L=%lf, N=%ld, v=%lf, t=%lf \n", L, N, v, t);
    double *S = malloc(sizeof(double)*N);
    double *k = malloc(sizeof(double)*N);
    if(!k||!S) {
         printf("Memory allocation error\n");
         return 1;
    }
    read_arrays(k, S, N);

    for (long i=0; i<N; i++) {
        k[i] = 1.0*k[i];
        S[i] = -1.0*S[i];
        printf("%ld %lf %lf \t",i,  k[i], S[i]);
    } printf("\n");



    band_mat bmat;
    long ncols = 2*N;
    long nbands_low = N;
    long nbands_up  = N;
    init_band_mat(&bmat, nbands_low, nbands_up, ncols);
    double *x = malloc(sizeof(double)*ncols);
    double *b = malloc(sizeof(double)*ncols);
    for (int i=0; i<ncols;i++){
        b[i]=0;
        x[i]=0;
    }

    long i;
    double dx = L/N;
    double dx2 = dx*dx;
    /* printf("%f, %f \n", dx, k[0]/dx); */
    // Boundary at A(0) and B(0)
    setv(&bmat, 0, 0, -k[0]/dx -v);
    setv(&bmat, 0, 1, k[0]/dx);
    setv(&bmat, N, N, -k[0]/dx -v);
    setv(&bmat, N, N+1, k[0]/dx);
    b[0] = S[0];
    /* printf("%lf, %lf \n", -k[0]/dx -v, k[0]/dx); */

    double a1, a2, a3;
    for(i=1; i<N; i++) {
        a1 = -(k[i]-k[i-1])/dx2 + k[i]/dx2 + v/dx;
        a2 = (k[i]-k[i-1])/dx2 -2*k[i]/dx2 - v/dx - t;
        a3 = k[i]/dx2;
        
        setv(&bmat, i, i-1, a1);
        setv(&bmat,i,i, a2);
        /* printf("i's in A: %ld \n", i); */
        if (i<N-1) {
            setv(&bmat,i,i+1,a3);
        }

        b[i] = S[i];   
    }

    double b1, b2, b3;
    for(i=N+1; i<ncols; i++) {
        i = i-N;
        b1 = -(k[i]-k[i-1])/dx2 + k[i]/dx2 + v/dx;
        b2 = (k[i]-k[i-1])/dx2 -2*k[i]/dx2 - v/dx;
        b3 = k[i]/dx2;
        i = i+N;

        setv(&bmat, i, i-1, b1);
        setv(&bmat, i, i, b2);
        /* printf("i's in B: %ld \n", i); */
        if (i<ncols-1) {
            setv(&bmat, i, i+1, b3);
        }
        b[i] = 0;   
    }
    // setting up the matrix (tI) in the lower left corner
    for (long j=0;j<N; j++){
        /* printf("(%ld, %ld, %lf) \n", i, j, t); */
        setv(&bmat, j+N, j, t);
    }

    /*  Print matrix for debugging: */ 
    if (0) {
        printmat(&bmat);            
    }

    printf("%d\n \n", solve_Ax_eq_b(&bmat, x, b));

    for (i=0; i<N;i++) {
        printf("%ld %lf \n", i, getv(&bmat, i+N, i));
    }

    for (i=0; i<2*N; i++) {
        printf("%ld %lf \n", i, getv(&bmat, i, i));
    }
    for (i=0; i<2*N-1; i++) {
        printf("%ld %lf \n", i, getv(&bmat, i, i+1));
    }

    //open output file
    FILE *out_file;
    out_file = fopen("output.txt", "w");
    if (out_file == NULL)
    {
        printf("Failed to open output file!\n");
        exit(1);
    }

    double xx;
    for(i=0; i<N; i++) {
        xx = i*dx;
        if (i == N-1) {
            fprintf(out_file, "%g %g %g \n", L, 0.0, 0.);
            printf("%g %g %g \n", L, 0.0, 0.0);
        } else {
            fprintf(out_file, "%g %g %g \n",xx,x[i],x[i+N]);
            printf("%g %g %g \n",xx,x[i],x[i+N]);
        }
    }

    return 0;
}


// {{{ helper functions

void read_params(double *L, long *N, double *v, double *t) {
   FILE *infile;
   if(!(infile=fopen("input.txt","r"))) {
       printf("Error opening file\n");
       exit(1);
   }
   if(4!=fscanf(infile,"%lf %ld %lf %lf",L, N ,v ,t)) {
       printf("Error reading parameters from file\n");
       exit(1);
   }
   fclose(infile);
}

void read_arrays(double *k, double *S, long N) {
   FILE *infile;
   if(!(infile=fopen("coefficients.txt","r"))) {
       printf("Error opening file\n");
       exit(1);
   }
    char buf[N];
    long i = 0;
    while (fgets(buf, sizeof buf, infile) != NULL) {
        // process line here
        if (sscanf(buf, "%lf %lf", &k[i], &S[i]) < 2) {
            printf("only accepts lines with two elements");
        }
        i+=1;
    }
   fclose(infile);
}

// }}}


