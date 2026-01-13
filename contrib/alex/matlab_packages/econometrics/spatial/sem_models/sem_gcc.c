
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <malloc.h>
#include <mex.h>
#include "randomlib.h"
#include "matrixjpl.h"

// *****************************************************
// mex sem_g file

// routine to integrate rho in the case of a diffuse prior on beta, homoscedastic model
//	int_rho(rhodet,detval,y,x,Wy,Wx,ngrid,n,k,den,z);

void    int_rho(double *rvec,
				double *ldet,
				double *y,
				double *Wy,
				double **x,
				double **Wx,
				int nrho,
				int n,
				int k,
     			double *den, // return argument
     			double *z)   // return argument
{

	int i, j, m, invt;
	double adj, nmk, nmk2;
	double *s, *num;
	double **xs, **xt, **xpx,*xpy, *ys, *b, *xb;
	double dsum;
	double tmp;
	

// compute denominator (integrating constant)
	      s    = dvector(0,nrho-1);
	      num  = dvector(0,nrho-1);
		  xs   = dmatrix(0,n-1,0,k-1);
		  xt   = dmatrix(0,k-1,0,n-1);
		  xpx  = dmatrix(0,k-1,0,k-1);
		  xpy  = dvector(0,k-1);
		  ys   = dvector(0,n-1);
		  b    = dvector(0,k-1);
		  xb   = dvector(0,n-1);


		  nmk = (double)(n-k);
		  nmk2 = nmk/2;	  

for(i=0; i<nrho; i++){
s[i] = 0.0;
	for(j=0; j<n; j++){
	ys[j] = 0.0;
	ys[j] = y[j] - rvec[i]*Wy[j];
		  for(m=0; m<k; m++){
		  b[m] = 0.0;
		  xs[j][m] = 0.0;
			  xs[j][m] = x[j][m] - rvec[i]*Wx[j][m];
		  }
	}  // end of for j loop
				  
	  transpose(xs,n,k,xt);  
      matmat(xt, k, n, xs, k, xpx);    
	  matvec(xt, k, n, ys, xpy);
      // compute inv(xs'*xs)
	  invt = inverse(xpx, k);
	  if (invt != 1)
		 mexPrintf("sem_gcc: Inversion error in rho integration \n");

	  // find bhat
	  matvec(xpx, k, k, xpy, b);
	  tmp = 0.0;
	  for(j=0; j<n; j++){
	  xb[j] = 0.0;
	  for(m=0; m<k; m++){
	  xb[j] = xb[j] + xs[j][m]*b[m];
	  }
      tmp = tmp + (ys[j] - xb[j])*(ys[j] - xb[j]);
	  } // end of for j loop
	  s[i] = ldet[i] - nmk2*log(tmp);
}

// adjustment for scaling that subtracts the maximum value
	adj = s[0];
	for(i=1; i<nrho; i++){
		if (s[i] > adj)
			adj = s[i];
	}
	
	dsum = 0.0;
	for(i=0; i<nrho; i++){
		den[i] = (s[i] - adj);
		den[i] = exp(den[i]);
		}

	dsum = 0.0; // trapezoid rule
    for(i=0; i<nrho-1; i++){
		dsum = dsum + (rvec[i+1]+rvec[i])*(den[i+1]-den[i])/2;
	}

		for(i=0; i<nrho; i++){
		z[i] = dabs(den[i]/dsum); // normalize rho post
		if (i == 0){
		den[i] = z[i];
		} else {
		den[i] = den[i-1] + z[i]; // cumulative sum
		}
	}
	

		
	free_dvector(s,0);
	free_dvector(num,0);
	free_dvector(xpy,0);
	free_dvector(ys,0);
	free_dvector(b,0);
	free_dvector(xb,0);

	free_dmatrix(xs,0,n-1,0);
	free_dmatrix(xt,0,k-1,0);
	free_dmatrix(xpx,0,k-1,0);


}



// routine to draw rho from the conditional posterior distribution
//	rho = draw_rho(rhodet,detval,y,x,Wy,Wx,v,ngrid,n,k,rho);

double draw_rho(double *rvec,
				double *ldet,
				double *y,
				double *Wy,
				double **x,
				double **Wx,
				double *v,
				int nrho,
				int n,
				int k,
     			double rho)
{

	int i, j, m, invt;
	double adj, nmk, nmk2, rnd;
	double *s, *den, *num;
	double **xs, **xt, **xpx,*xpy, *ys, *b, *xb;
	double dsum, rsum;
	double tmp, vsqrt;
	

// compute denominator (integrating constant)
	      s    = dvector(0,nrho-1);
	      den  = dvector(0,nrho-1);
	      num  = dvector(0,nrho-1);
		  xs   = dmatrix(0,n-1,0,k-1);
		  xt   = dmatrix(0,k-1,0,n-1);
		  xpx  = dmatrix(0,k-1,0,k-1);
		  xpy  = dvector(0,k-1);
		  ys   = dvector(0,n-1);
		  b    = dvector(0,k-1);
		  xb   = dvector(0,n-1);


		  nmk = (double)(n-k);
		  nmk2 = nmk/2;	  

for(i=0; i<nrho; i++){
s[i] = 0.0;
	for(j=0; j<n; j++){
	ys[j] = 0.0;
	vsqrt = 1.0/sqrt(v[j]);
	ys[j] = y[j] - rvec[i]*Wy[j];
	ys[j] = ys[j]*vsqrt;
		  for(m=0; m<k; m++){
		  b[m] = 0.0;
		  xs[j][m] = 0.0;
			  xs[j][m] = x[j][m] - rvec[i]*Wx[j][m];
			  xs[j][m] = xs[j][m]*vsqrt;
		  }
	}  // end of for j loop
				  
	  transpose(xs,n,k,xt);  
      matmat(xt, k, n, xs, k, xpx);    
	  matvec(xt, k, n, ys, xpy);
      // compute inv(xs'*xs)
	  invt = inverse(xpx, k);
	  if (invt != 1)
		 mexPrintf("sem_gcc: Inversion error in beta conditional \n");

	  // find bhat
	  matvec(xpx, k, k, xpy, b);
	  tmp = 0.0;
	  for(j=0; j<n; j++){
	  xb[j] = 0.0;
	  for(m=0; m<k; m++){
	  xb[j] = xb[j] + xs[j][m]*b[m];
	  }
      tmp = tmp + (ys[j] - xb[j])*(ys[j] - xb[j]);
	  } // end of for j loop
	  s[i] = ldet[i] - nmk2*log(tmp);
}

// adjustment for scaling that subtracts the maximum value
	adj = s[0];
	for(i=1; i<nrho; i++){
		if (s[i] > adj)
			adj = s[i];
	}
	
	dsum = 0.0;
	for(i=0; i<nrho; i++){
		den[i] = (s[i] - adj);
		den[i] = exp(den[i]);
		}

	dsum = 0.0; // trapezoid rule
    for(i=0; i<nrho-1; i++){
		dsum = dsum + (rvec[i+1]+rvec[i])*(den[i+1]-den[i])/2;
	}

	    rsum = 0.0;
		for(i=0; i<nrho; i++){
		s[i] = dabs(den[i]/dsum); // normalize rho post
        rsum = rsum + s[i];
		if (i == 0){
		den[i] = s[i];
		} else {
		den[i] = den[i-1] + s[i]; // cumulative sum
		}
	}
	
    rnd = rsum*ranf();

	// create rho draw via inversion
	for(i=0; i<nrho; i++){
	  if (rnd <= den[i]){
	  rho = rvec[i];
	  break;
	  }
	  }
		
		
	free_dvector(s,0);
	free_dvector(den,0);
	free_dvector(num,0);
	free_dvector(xpy,0);
	free_dvector(ys,0);
	free_dvector(b,0);
	free_dvector(xb,0);

	free_dmatrix(xs,0,n-1,0);
	free_dmatrix(xt,0,k-1,0);
	free_dmatrix(xpx,0,k-1,0);

	return (rho);


}


// routine to draw rho from the conditional posterior distribution
// in the case of a homoscedastic prior and diffuse prior for beta

//     rho = draw_rhoh(rvec,den,zz,nrho,rho);
double draw_rhoh(double *rvec,
				double *den,
				double *zz,
				int nrho,
     			double rho)
{
double rsum, rnd;
int i;

    rsum = 0.0;
    for(i=0; i<nrho; i++)
    rsum = rsum + zz[i];

    rnd = rsum*ranf();

	// create rho draw via inversion
	for(i=0; i<nrho; i++){
	  if (rnd <= den[i]){
	  rho = rvec[i];
	  break;
	  }
	  }
		
	return (rho);

}



// *****************************************************
// main sampler

// estimates robust spatial error model using MCMC
//    sem_gc(pdraw,bdraw,sdraw,rdraw,vmean,y,Wy,x,Wx,ldet,n,k,ngrid,rval,ndraw,nomit,nu,d0,mm,kk,TI,TIc,rho0,sig0,priorb);

void sem_gc(
			// Output arguments
			double *pdraw, // pout = draws for rho (ndraw x 1) vector
            double *bdraw, // bout = draws for beta (ndraw,k) matrix
            double *sdraw, // sout = draws for sige (ndraw,1) vector
            double *rdraw, // rout = draws for rval if mm .ne. 0
            double *vmean, // vmean = mean of vi draws (n,1) vector

			// Input arguments
            double *y,     // y = nx1 lhs vector
 			double *Wy,    // Wy = nx1  weight matrix times y-vector     
            double *x,     // x = nxk explanatory variables matrix 
			double *Wx,    // Wx = nxk weight matrix times x-matrix
            double *ldet,  // ldet = ngrid x 2 matrix with [rho , log det values]
            int n,         // n = # of observations
			int k,         // k = # of explanatory variables
            int ngrid,     // ngrid = # of values in lndet (rows)
            double rval,   // rval = hyperparameter r
            int ndraw,     // ndraw = # of draws
            int nomit,     // nomit = # of burn-in draws to omit
            double nu,     // nu = gamma prior for sige
            double d0,     // d0 = gamma prior for sige
            double mm,     // mm = exp prior for rval
            double kk,     // kk = exp prior for rval
			double *TI,    // prior var-cov for beta (inverted in matlab)
			double *TIc,   // prior var-cov * prior mean
			double rho0,   // starting value for rho 
			double sig0,   // starting value for sige
			int priorb)    // flag for diffuse prior on beta
{
    // local stuff
    int i, j, iter;
    double *ys, *yss, **xs, **xt, **xmat, **Wxmat, **xss, **xpx, *xpy, *v;
	double **priorv, *priorm, *rhodet, *detval;
	double *bhat, *btmp, *bnorm, **covm, *xb, **xpxix;
    double rho, rmin, rmax, epe, chisq;
    double sige, vsqrt, dof, evec, chiv;
	BOOL invt;
	   
    // allocate vectors
	xt   = dmatrix(0,k-1,0,n-1);
	xs   = dmatrix(0,n-1,0,k-1);
	xpx  = dmatrix(0,k-1,0,k-1);
	covm = dmatrix(0,k-1,0,k-1);
	xmat = dmatrix(0,n-1,0,k-1);
	Wxmat = dmatrix(0,n-1,0,k-1);
	xss   = dmatrix(0,n-1,0,k-1);
	priorv = dmatrix(0,k-1,0,k-1);
	xpxix = dmatrix(0,k-1,0,n-1);
	
	ys   = dvector(0,n-1);
	yss  = dvector(0,n-1);
    v    = dvector(0,n-1);
	xpy  = dvector(0,k-1);
	priorm = dvector(0,k-1);
	bhat   = dvector(0,k-1);
	btmp   = dvector(0,k-1);
	bnorm  = dvector(0,k-1);
	xb     = dvector(0,n-1);
	rhodet = dvector(0,ngrid-1);
	detval = dvector(0,ngrid-1);

	
	// put x into xmat
	// #define X(i,j) x[i + j*n]

	for(i=0; i<n; i++){
		v[i] = 1.0;
		for(j=0; j<k; j++){
			xmat[i][j] = x[i + j*n];
			Wxmat[i][j] = Wx[i + j*n];
		}
	}
	
	
	// put rho into rhodet vector
	// #define X(i,j) x[i + j*n]

	for(i=0; i<ngrid; i++){
	    j=0;
		rhodet[i] = ldet[i + j*ngrid];
		j=1;
		detval[i] = ldet[i + j*ngrid];
		}
		

    // put TI, TIc prior info into priorm, priorv
	for(i=0; i<k; i++){
		priorm[i] = TIc[i];
	    for(j=0; j<k; j++)
			priorv[i][j] = TI[i + j*k];
	}
    
   
// initializations

    dof = ((double) n + 2.0*nu);    
    sige = sig0;	
    rmin = rhodet[0];
    rmax = rhodet[ngrid-1];   
    rho = rho0;
    evec = 0.0;
 
// do MCMC draws on rho, sige, V

// ======================================
// start the sampler
// ======================================

for(iter=0; iter<ndraw; iter++){

// apply variance scalars using matmulc
	for(i=0; i<n; i++){
		vsqrt = 1.0/sqrt(v[i]);
		ys[i] = y[i] - rho*Wy[i];
		yss[i] = ys[i]*vsqrt;

		for(j=0; j<k; j++){
		xs[i][j] = xmat[i][j] - rho*Wxmat[i][j];
		xss[i][j] = xs[i][j]*vsqrt;
		}
	}


// ==================================================
// update beta with a multivariate normal draw
	transpose(xss,n,k,xt);
   
    matmat(xt, k, n, xss, k, xpx);
    
	matvec(xt, k, n, yss, xpy);

// add prior information
	for(i=0; i<k; i++){
		xpy[i] = xpy[i] + sige*priorm[i];
		for(j=0; j<k; j++)
			xpx[i][j] = xpx[i][j] + sige*priorv[i][j];
	}

// invert xpx
     invt = inverse(xpx, k);
	 if (invt != 1)
		 mexPrintf("sem_gcc: Inversion error in beta conditional \n");
		 
// find bhat
	 matvec(xpx, k, k, xpy, btmp);

// multiply xpx-inverse times sige
     for(i=0; i<k; i++){
     for(j=0; j<k; j++)
     covm[i][j] = xpx[i][j]*sige;
     }
     
 // do multivariate normal draw based on sige*xpx-inverse
	 normal_rndc(covm, k, bnorm);

	for(i=0; i<k; i++){
	bhat[i] = 0.0;
	bhat[i] = btmp[i] + bnorm[i];
	}

// ==================================================
// update sigma with a chi-squared draw

	// form xss*bhat
	matvec(xss, n, k, bhat, xb);

    epe = 0.0;
	for(i=0; i<n; i++){
		evec = yss[i] - xb[i];
		epe = epe + evec*evec;
	}
	chisq = genchi(dof);
    sige = (2*d0 + epe)/chisq;
    
// ==================================================
// update vi with a chi-squared draw

	// form xs*bhat
	 matvec(xs, n, k, bhat, xb);

	 for(i=0; i<n; i++){
	 epe = 0.0;
	 evec = 0.0;
     evec = ys[i] - xb[i];
	 chiv = genchi(rval+1.0);
	 epe = (evec*evec)/sige;
	 v[i] = (epe + rval)/chiv;
	 }

// ==================================================
// update rho using numerical integration
	rho = draw_rho(rhodet,detval,y,Wy,xmat,Wxmat,v,ngrid,n,k,rho);
    
//==================================================
// update rval if mm .ne. 0
    if (mm != 0.0){
    rval = gengam(mm,kk);
    }
    

// ==================================================
// save the draws
   *(pdraw+iter) = rho;
   *(sdraw+iter) = sige;
   if(mm != 0.0)
   *(rdraw+iter) = rval;	

   
// #define X(i,j) x[i + j*n]
   for(j=0; j<k; j++)
   bdraw[iter + j*ndraw] = bhat[j];
   
   *(rdraw+iter) = rval;
   if (iter > nomit-1){
		for(i=0; i<n; i++)
        vmean[i] = vmean[i] + v[i]/((double) (ndraw-nomit));
   }
        
}
// ======================================
// end of the sampler
// ======================================


// free up allocated vectors
	free_dmatrix(xt,0,k-1,0);
	free_dmatrix(xs,0,n-1,0);
	free_dmatrix(xpx,0,k-1,0);
	free_dmatrix(xmat,0,n-1,0);
	free_dmatrix(Wxmat,0,n-1,0);
	free_dmatrix(xss,0,n-1,0);
	free_dmatrix(priorv,0,k-1,0);
	free_dmatrix(covm,0,k-1,0);
	free_dmatrix(xpxix,0,k-1,0);

	free_dvector(xpy,0);
    free_dvector(ys,0);
    free_dvector(yss,0);
    free_dvector(v,0);
	free_dvector(bhat,0);
	free_dvector(bnorm,0);
	free_dvector(btmp,0);
	free_dvector(priorm,0);
	free_dvector(xb,0);
	free_dvector(rhodet,0);
	free_dvector(detval,0);



} // end of sem_gc

// estimates homoscedastic spatial error model using MCMC
//    sem_gch(pdraw,bdraw,sdraw,y,Wy,x,Wx,ldet,n,k,ngrid,ndraw,nomit,nu,d0,TI,TIc,rho0,sig0,priorb);

void sem_gch(
			// Output arguments
			double *pdraw, // pout = draws for rho (ndraw x 1) vector
            double *bdraw, // bout = draws for beta (ndraw,k) matrix
            double *sdraw, // sout = draws for sige (ndraw,1) vector
 
			// Input arguments
            double *y,     // y = nx1 lhs vector
 			double *Wy,    // Wy = nx1  weight matrix times y-vector     
            double *x,     // x = nxk explanatory variables matrix 
			double *Wx,    // Wx = nxk weight matrix times x-matrix
            double *ldet,  // ldet = ngrid x 2 matrix with [rho , log det values]
            int n,         // n = # of observations
			int k,         // k = # of explanatory variables
            int ngrid,     // ngrid = # of values in lndet (rows)
            int ndraw,     // ndraw = # of draws
            int nomit,     // nomit = # of burn-in draws to omit
            double nu,     // nu = gamma prior for sige
            double d0,     // d0 = gamma prior for sige
     		double *TI,    // prior var-cov for beta (inverted in matlab)
			double *TIc,   // prior var-cov * prior mean
			double rho0,   // starting value for rho 
			double sig0,   // starting value for sige
			int priorb)    // flag for diffuse prior  
{
    // local stuff
    int i, j, iter;
    double *ys, **xs, **xt, **xmat, **Wxmat, **xpx, *xpy;
	double **priorv, *priorm, *rhodet, *detval, *den, *zz, *v;
	double *bhat, *btmp, *bnorm, **covm, *xb, **xpxix;
    double rho, rmin, rmax, epe, chisq;
    double sige, dof, evec;
	BOOL invt;
	   
    // allocate vectors
	xt   = dmatrix(0,k-1,0,n-1);
	xs   = dmatrix(0,n-1,0,k-1);
	xpx  = dmatrix(0,k-1,0,k-1);
	covm = dmatrix(0,k-1,0,k-1);
	xmat = dmatrix(0,n-1,0,k-1);
	Wxmat = dmatrix(0,n-1,0,k-1);
	priorv = dmatrix(0,k-1,0,k-1);
	xpxix = dmatrix(0,k-1,0,n-1);
	
	ys   = dvector(0,n-1);
	v    = dvector(0,n-1);
	xpy  = dvector(0,k-1);
	priorm = dvector(0,k-1);
	bhat   = dvector(0,k-1);
	btmp   = dvector(0,k-1);
	bnorm  = dvector(0,k-1);
	xb     = dvector(0,n-1);
	rhodet = dvector(0,ngrid-1);
	detval = dvector(0,ngrid-1);
	den    = dvector(0,ngrid-1);
	zz     = dvector(0,ngrid-1);


	// put x into xmat
	// #define X(i,j) x[i + j*n]

	for(i=0; i<n; i++){
	v[i] = 1.0;
		for(j=0; j<k; j++){
			xmat[i][j] = x[i + j*n];
			Wxmat[i][j] = Wx[i + j*n];
		}
	}
	
	// put rho into rhodet vector
	// #define X(i,j) x[i + j*n]

	for(i=0; i<ngrid; i++){
	    j=0;
		rhodet[i] = ldet[i + j*ngrid];
		j=1;
		detval[i] = ldet[i + j*ngrid];
		}
		

    // put TI, TIc prior info into priorm, priorv
	for(i=0; i<k; i++){
		priorm[i] = TIc[i];
	    for(j=0; j<k; j++)
			priorv[i][j] = TI[i + j*k];
	}
    
   
// initializations

    dof = ((double) n + 2.0*nu);    
    sige = sig0;	
    rmin = rhodet[0];
    rmax = rhodet[ngrid-1];   
    rho = rho0;
    evec = 0.0;
 
// if we have a diffuse prior on beta integrate rho up front
if (priorb == 0){
  int_rho(rhodet,detval,y,Wy,xmat,Wxmat,ngrid,n,k,den,zz);
  // return arguments are den and zz
  }


// do MCMC draws on rho, sige, V

// ======================================
// start the sampler
// ======================================

for(iter=0; iter<ndraw; iter++){

// quasi-differencing
	for(i=0; i<n; i++){
		ys[i] = y[i] - rho*Wy[i];

		for(j=0; j<k; j++){
		xs[i][j] = xmat[i][j] - rho*Wxmat[i][j];
		}
	}


// ==================================================
// update beta with a multivariate normal draw
	transpose(xs,n,k,xt);
   
    matmat(xt, k, n, xs, k, xpx);
    
	matvec(xt, k, n, ys, xpy);

// add prior information
	for(i=0; i<k; i++){
		xpy[i] = xpy[i] + sige*priorm[i];
		for(j=0; j<k; j++)
			xpx[i][j] = xpx[i][j] + sige*priorv[i][j];
	}

// invert xpx
     invt = inverse(xpx, k);
	 if (invt != 1)
		 mexPrintf("sem_gcc: Inversion error in beta conditional \n");
		 
// find bhat
	 matvec(xpx, k, k, xpy, btmp);

// multiply xpx-inverse times sige
     for(i=0; i<k; i++){
     for(j=0; j<k; j++)
     covm[i][j] = xpx[i][j]*sige;
     }
     
 // do multivariate normal draw based on sige*xpx-inverse
	 normal_rndc(covm, k, bnorm);

	for(i=0; i<k; i++){
	bhat[i] = 0.0;
	bhat[i] = btmp[i] + bnorm[i];
	}

// ==================================================
// update sigma with a chi-squared draw

	// form xs*bhat
	matvec(xs, n, k, bhat, xb);

    epe = 0.0;
	for(i=0; i<n; i++){
		evec = ys[i] - xb[i];
		epe = epe + evec*evec;
	}
	chisq = genchi(dof);
    sige = (2*d0 + epe)/chisq;
    

// ==================================================
// update rho using numerical integration

// if we have a diffuse prior
    if (priorb == 0) {
    rho = draw_rhoh(rhodet,den,zz,ngrid,rho);
    }else{
	rho = draw_rho(rhodet,detval,y,Wy,xmat,Wxmat,v,ngrid,n,k,rho);
    }

// ==================================================
// save the draws
   *(pdraw+iter) = rho;
   *(sdraw+iter) = sige;
   
// #define X(i,j) x[i + j*n]
   for(j=0; j<k; j++)
   bdraw[iter + j*ndraw] = bhat[j];
   
        
}
// ======================================
// end of the sampler
// ======================================


// free up allocated vectors
	free_dmatrix(xt,0,k-1,0);
	free_dmatrix(xs,0,n-1,0);
	free_dmatrix(xpx,0,k-1,0);
	free_dmatrix(xmat,0,n-1,0);
	free_dmatrix(Wxmat,0,n-1,0);
	free_dmatrix(priorv,0,k-1,0);
	free_dmatrix(covm,0,k-1,0);
	free_dmatrix(xpxix,0,k-1,0);

	free_dvector(xpy,0);
	free_dvector(v,0);
    free_dvector(ys,0);
	free_dvector(bhat,0);
	free_dvector(bnorm,0);
	free_dvector(btmp,0);
	free_dvector(priorm,0);
	free_dvector(xb,0);
	free_dvector(rhodet,0);
	free_dvector(detval,0);


} // end of sem_gch



void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[] )
{
  double *y, *Wy, *Wx, *x, *ldet, *pdraw, *bdraw, *sdraw, *rdraw, *vmean;
  int  n, k, ngrid, ndraw, nomit, priorb;
  double rval, nu, d0, mm, kk;
  double *TI, *TIc, c, T, rho0, sig0;
  long *Seed1, *Seed2;
  int buflen, status, flag;
  char *phrase;
  static char phrase2[8];
  
    phrase2[0] = 'h';
    phrase2[1] = 'h';
    phrase2[2] = 'i';
    phrase2[3] = 't';
    phrase2[4] = 'h';
    phrase2[5] = 'e';
    phrase2[6] = 'r';
    phrase2[7] = 'e';



  /* Check for proper number of arguments. */
  if(nrhs == 17) {
     flag = 0;
    phrase = phrase2;
  } else if (nrhs == 18){
    flag = 1;
  } else {
    mexErrMsgTxt("sem_gcc: 17 or 18 inputs required.");
  }

  if(nlhs != 5) {
    mexErrMsgTxt("sem_gcc: 5 output arguments needed");
  }

    if (flag == 1) {
    // input must be a string
    if ( mxIsChar(prhs[17]) != 1)
      mexErrMsgTxt("sem_gcc: seed must be a string.");
    // input must be a row vector
    if (mxGetM(prhs[17])!=1)
      mexErrMsgTxt("sem_gcc: seed input must be a row vector.");

    // get the length of the input string
    buflen = (mxGetM(prhs[17]) * mxGetN(prhs[17])) + 1;

    // allocate memory for input string
    phrase = mxCalloc(buflen, sizeof(char));

    // copy the string data from prhs[0] into a C string input_ buf.
    // If the string array contains several rows, they are copied,
    // one column at a time, into one long string array.
    //
    status = mxGetString(prhs[17], phrase, buflen);
    if(status != 0)
      mexWarnMsgTxt("sem_gcc: Not enough space. seed string truncated.");
    }
    
    // allow the user to set a seed or rely on clock-based seed
   if (flag == 0) {
    setSeedTimeCore(phrase);
   } else {
	phrtsd(phrase,&Seed1,&Seed2);
    setall(Seed1,Seed2);
   }

  	//  parse input arguments
    //[rout,bout,sout,rdraw,vmean] = ...
    // sem_gcc(y,x,Wy,Wx,detval,rval,ndraw,nomit,nu,d0,mm,kk,TI,TIc,p0,sig0);


     y = mxGetPr(prhs[0]);
     x = mxGetPr(prhs[1]);
     n = mxGetM(prhs[1]);
     k = mxGetN(prhs[1]);
	 Wy = mxGetPr(prhs[2]);
	 Wx = mxGetPr(prhs[3]);

	 ldet = mxGetPr(prhs[4]);
	 ngrid = mxGetM(prhs[4]);
	 rval = mxGetScalar(prhs[5]);
	 ndraw = (int) mxGetScalar(prhs[6]);
	 nomit = (int) mxGetScalar(prhs[7]);
	 nu = mxGetScalar(prhs[8]);
	 d0 = mxGetScalar(prhs[9]);
	 mm = mxGetScalar(prhs[10]);
	 kk = mxGetScalar(prhs[11]);
	 TI = mxGetPr(prhs[12]);
	 TIc = mxGetPr(prhs[13]);
	 rho0 = mxGetScalar(prhs[14]);
	 sig0 = mxGetScalar(prhs[15]);
	 priorb = (int) mxGetScalar(prhs[16]);
	 
	// no need for error checking on inputs
	// since this was done in the matlab function


    /* Create matrices for the return arguments */
	 //bdraw,sdraw,rdraw,vmean
    plhs[0] = mxCreateDoubleMatrix(ndraw,1, mxREAL); // rho draws
	plhs[1] = mxCreateDoubleMatrix(ndraw,k, mxREAL); // beta draws
	plhs[2] = mxCreateDoubleMatrix(ndraw,1, mxREAL); // sige draws
	plhs[3] = mxCreateDoubleMatrix(ndraw,1, mxREAL); // rval draws
	plhs[4] = mxCreateDoubleMatrix(n,1, mxREAL);     // vmean

    pdraw = mxGetPr(plhs[0]);
	bdraw = mxGetPr(plhs[1]);
	sdraw = mxGetPr(plhs[2]);
	rdraw = mxGetPr(plhs[3]);
	vmean = mxGetPr(plhs[4]);


    /* Call the  subroutine. */
    if (rval != 0){
    sem_gc(pdraw,bdraw,sdraw,rdraw,vmean,y,Wy,x,Wx,ldet,n,k,ngrid,rval,ndraw,nomit,nu,d0,mm,kk,TI,TIc,rho0,sig0,priorb);
    }else{
    sem_gch(pdraw,bdraw,sdraw,y,Wy,x,Wx,ldet,n,k,ngrid,ndraw,nomit,nu,d0,TI,TIc,rho0,sig0,priorb);
    }

}


