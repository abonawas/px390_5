#include <stdlib.h>
#include <stdio.h>
#include <lapacke.h>

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

// Diagnostic routines: set to 1 to enable.
#define DIAGS 1

/* An example of how to use the band matrix routines to solve a PDE:
   The equation solved is related to the steady state solution of the heat 
   diffusion equation.   
*/
int main() {
  band_mat bmat;
  double L= 1.0;
  long ncols = 20;
  double dx = L/(ncols +1);
  double dx2 = dx*dx;

  long nbands_low = 1;  
  long nbands_up  = 1;
  init_band_mat(&bmat, nbands_low, nbands_up, ncols);
  double *x = malloc(sizeof(double)*ncols);
  double *b = malloc(sizeof(double)*ncols);

    long i;
  /* Loop over the equation number and set the matrix
     values equal to the coefficients of the grid values 
     note boundaries treated with special cases           */
  for(i=0; i<ncols; i++) {
    if(i>0)       {setv(&bmat,i,i-1,-1.0/dx2);};
    setv(               &bmat,i,i,   -2.0/dx2);
    if(i<ncols-1) {setv(&bmat,i,i+1,1.0/dx2);};
    /* Uniform source term in heat equation */
    b[i] = -1.0;   
  }
  
  /*  Print matrix for debugging: */ 
  if (1) {
    printmat(&bmat);            
  }

  solve_Ax_eq_b(&bmat, x, b);

  for(i=0; i<ncols; i++) {
    printf("%ld %g %g \n",i,x[i],b[i]);
  }

}
