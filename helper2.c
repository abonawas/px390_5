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

    /* for (long i=0; i<N; i++) { */
    /*     k[i] = 1.0*k[i]; */
    /*     S[i] = -1.0*S[i]; */
    /*     /1* printf("%ld %lf %lf \t",i,  k[i], S[i]); *1/ */
    /* } printf("\n"); */


    band_mat bmat;
    long ncols = 2*N;
    long nbands_low = 2;
    long nbands_up  = 2;
    init_band_mat(&bmat, nbands_low, nbands_up, ncols);

    double *x = malloc(sizeof(double)*ncols);
    double *b = malloc(sizeof(double)*ncols);
    if(!x||!b) {
         printf("Memory allocation error\n");
         return 1;
    }

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
    setv(&bmat, 0, 2, k[0]/dx);
    setv(&bmat, 1, 1, -k[0]/dx -v);
    setv(&bmat, 1, 3, k[0]/dx);
    setv(&bmat, 1, 0, t);
    b[0] = -S[0];
    /* printf("%lf, %lf \n", -k[0]/dx -v, k[0]/dx); */

    double a1, a2, a3;
    double b1, b2, b3;
    long j =1;
    for(i=2; i<ncols; i+=2) {

        a1 = -(k[j]-k[j-1])/dx2 + k[j]/dx2 + v/dx;
        a2 = (k[j]-k[j-1])/dx2 -2*k[j]/dx2 - v/dx - t;
        a3 = k[j]/dx2;
        
        setv(&bmat, i, i-2, a1);
        setv(&bmat,i,i, a2);
        /* printf("i's in A: %ld \n", i); */
        if (i<ncols-2) {
            setv(&bmat,i,i+2,a3);
        }
        b[i] = -S[j];   
    
        b1 = -(k[j]-k[j-1])/dx2 + k[j]/dx2 + v/dx;
        b2 = (k[j]-k[j-1])/dx2 -2*k[j]/dx2 - v/dx;
        b3 = k[j]/dx2;

        setv(&bmat, i+1, i+1-2, b1);
        setv(&bmat, i+1, i+1, b2);
        /* printf("i's in B: %ld \n", i); */
        if (i<ncols-2) {
            setv(&bmat, i+1, i+1+2, b3);
        }
        setv(&bmat, i+1, i+1-1, t);
        b[i+1] = 0;   
        j+=1;
    }

    printf("%d\n \n", solve_Ax_eq_b(&bmat, x, b));

    /* for (i=1; i<ncols;i++) { */
    /*     printf("%ld %lf \n", i, getv(&bmat, i, i-1)); */
    /* } */

    /* for (i=0; i<2*N; i++) { */
    /*     printf("%ld %lf %lf \n", i, getv(&bmat, i, i), b[i]); */
    /* } */
    /* for (i=0; i<2*N-2; i++) { */
    /*     printf("%ld %lf \n", i, getv(&bmat, i, i+2)); */
    /* } */

    //open output file
    FILE *out_file;
    out_file = fopen("output.txt", "w");
    if (out_file == NULL)
    {
        printf("Failed to open output file!\n");
        exit(1);
    }

    double xx;
    j = 0;
    for(i=0; i<ncols; i+=2) {
        xx = j*dx;
        if (i == ncols-2) {
            fprintf(out_file, "%g %g %g \n", L, 0.0, 0.);
            printf("%g %g %g \n", L, 0.0, 0.0);
        } else {
            fprintf(out_file, "%g %g %g \n",xx,x[i],x[i+1]);
            printf("%g %g %g \n",xx,x[i],x[i+1]);
        }
        j+=1;
    }

    free(b);
    free(x);
    free(S);
    free(k);
    fclose(out_file);
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


