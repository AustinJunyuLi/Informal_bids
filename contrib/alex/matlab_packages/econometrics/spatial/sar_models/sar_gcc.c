
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <malloc.h>
#include <mex.h>
#include "randomlib.h"
#include "matrixjpl.h"

// *****************************************************
// mex sar_g file

// routine to draw rho from the conditional posterior distribution
//	rho = draw_rho(rhodet,detval,epe0,eped,epe0d,ngrid,n,k,rho);

double draw_rho(double *rvec,
				double *ldet,
				double epe0,
				double eped,
				double epe0d,
				int nrho,
				int n,
				int k,
     			double rho)
{

	int i;
	double adj, nmk, nmk2, rnd;
	double *s, *den, *num;
	double dsum, rsum, z;
	

// compute denominator (integrating constant)
	      s    = dvector(0,nrho-1);
	      den  = dvector(0,nrho-1);
	      num  = dvector(0,nrho-1);

		  nmk = (double)(n-k);
		  nmk2 = nmk/2;	  

	  for(i=0; i<nrho; i++){
		      z = epe0 - 2.0*rvec[i]*epe0d + rvec[i]*rvec[i]*eped;
              s[i] = ldet[i] - nmk2*log(z);
	  }

// adjustment for scaling that subtracts the maximum value
	adj = s[0];
	for(i=1; i<nrho; i++){
		if (s[i] > adj)
			adj = s[i];
	}
	
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

	return (rho);

}


// *****************************************************
// main sampler for heteroscedastic model

// estimates robust spatial autoregressive model using MCMC
//    sar_gc(pdraw,bdraw,sdraw,rdraw,vmean,y,x,Wy,ldet,n,k,ngrid,rval,ndraw,nomit,nu,d0,mm,kk,TI,TIc,rho0,sig0);

void sar_gc(
			// Output arguments
			double *pdraw, // pout = draws for rho (ndraw x 1) vector
            double *bdraw, // bout = draws for beta (ndraw,k) matrix
            double *sdraw, // sout = draws for sige (ndraw,1) vector
            double *rdraw, // rout = draws for rval if mm .ne. 0
            double *vmean, // vmean = mean of vi draws (n,1) vector

			// Input arguments
            double *y,     // y = nx1 lhs vector
 			double *x,     // x = nxk explanatory variables matrix           
            double *Wy,    // Wy = nx1 weight matrix times y-vector
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
			double sig0)   // starting value for sige
{
    // local stuff
    int i, j, iter, pflag;
    double *ys, *yss, **xs, *Wys, **xt, **xmat, **xpx, *xpy, *v;
	double **priorv, *priorm, *rhodet, *detval;
	double *bhat, *btmp, *bnorm, **covm, *xb, **xpxix;
    double rho, rmin, rmax, p, epe, chisq;
    double sige, vsqrt, dof, evec, chiv;
	double *e0, *ed, *b0, *bd, *xb0, *xbd, epe0, eped, epe0d;
	BOOL invt;
	   
    // allocate vectors
	xt   = dmatrix(0,k-1,0,n-1);
	xs   = dmatrix(0,n-1,0,k-1);
	xpx  = dmatrix(0,k-1,0,k-1);
	covm = dmatrix(0,k-1,0,k-1);
	xmat = dmatrix(0,n-1,0,k-1);
	priorv = dmatrix(0,k-1,0,k-1);
	xpxix = dmatrix(0,k-1,0,n-1);
	
	ys   = dvector(0,n-1);
	yss  = dvector(0,n-1);
    Wys  = dvector(0,n-1);
    v    = dvector(0,n-1);
	xpy  = dvector(0,k-1);
	priorm = dvector(0,k-1);
	bhat   = dvector(0,k-1);
	btmp   = dvector(0,k-1);
	bnorm  = dvector(0,k-1);
	xb     = dvector(0,n-1);
	rhodet = dvector(0,ngrid-1);
	detval = dvector(0,ngrid-1);

	e0   = dvector(0,n-1);
    ed   = dvector(0,n-1);
    b0   = dvector(0,k-1);
	bd   = dvector(0,k-1);
	xb0  = dvector(0,n-1);
	xbd  = dvector(0,n-1);

	
	// put x into xmat
	// #define X(i,j) x[i + j*n]

	for(i=0; i<n; i++){
		v[i] = 1.0;
		for(j=0; j<k; j++)
			xmat[i][j] = x[i + j*n];
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
		ys[i] = y[i]*vsqrt;
		Wys[i] = Wy[i]*vsqrt;
		yss[i] = ys[i] - rho*Wys[i];
		for(j=0; j<k; j++)
		xs[i][j] = xmat[i][j]*vsqrt;
	}

// ==================================================
// update beta with a multivariate normal draw
	transpose(xs,n,k,xt);
   
    matmat(xt, k, n, xs, k, xpx);
    
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
		 mexPrintf("sar_gcc: Inversion error in beta conditional \n");
		 
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
		evec = ys[i] - rho*Wys[i] - xb[i];
		epe = epe + evec*evec;
	}
	chisq = genchi(dof);
    sige = (2*d0 + epe)/chisq;
    
// ==================================================
// update vi with a chi-squared draw

	// form x*bhat
	 matvec(xmat, n, k, bhat, xb);

	 for(i=0; i<n; i++){
	 epe = 0.0;
	 evec = 0.0;
     evec = y[i] - rho*Wy[i] - xb[i];
	 chiv = genchi(rval+1.0);
	 epe = (evec*evec)/sige;
	 v[i] = (epe + rval)/chiv;
	 }

// ==================================================
// update rho using numerical integration

          matmat(xpx, k, k, xt, n, xpxix);
		  matvec(xpxix, k, n, ys,  b0);
		  matvec(xpxix, k, n, Wys, bd);	
		  matvec(xs, n, k, b0, xb0);	
          matvec(xs, n, k, bd, xbd);

          for(i=0; i<n; i++){
		  e0[i] = ys[i]  - xb0[i];
		  ed[i] = Wys[i] - xbd[i];
		  }

		  epe0 = 0.0;
		  eped = 0.0;
		  epe0d = 0.0;
		  for(i=0; i<n; i++){
			  epe0 = epe0 + e0[i]*e0[i];
			  eped = eped + ed[i]*ed[i];
			  epe0d = epe0d + ed[i]*e0[i];
		  }
	
	rho = draw_rho(rhodet,detval,epe0,eped,epe0d,ngrid,n,k,rho);
    
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
	free_dmatrix(priorv,0,k-1,0);
	free_dmatrix(covm,0,k-1,0);
	free_dmatrix(xpxix,0,k-1,0);

	free_dvector(xpy,0);
    free_dvector(ys,0);
    free_dvector(yss,0);
	free_dvector(Wys,0);
    free_dvector(v,0);
	free_dvector(bhat,0);
	free_dvector(bnorm,0);
	free_dvector(btmp,0);
	free_dvector(priorm,0);
	free_dvector(xb,0);
	free_dvector(rhodet,0);
	free_dvector(detval,0);

	free_dvector(e0,0);
	free_dvector(ed,0);
	free_dvector(b0,0);
	free_dvector(bd,0);
	free_dvector(xb0,0);
	free_dvector(xbd,0);




} // end of sar_gc


// *****************************************************
// main sampler for homoscedastic model

// estimates homoscedastic spatial autoregressive model using MCMC
//    sar_gch(pdraw,bdraw,sdraw,y,x,Wy,ldet,n,k,ngrid,ndraw,nomit,nu,d0,TI,TIc,rho0,sig0);

void sar_gch(
			// Output arguments
			double *pdraw, // pout = draws for rho (ndraw x 1) vector
            double *bdraw, // bout = draws for beta (ndraw,k) matrix
            double *sdraw, // sout = draws for sige (ndraw,1) vector

			// Input arguments
            double *y,     // y = nx1 lhs vector
 			double *x,     // x = nxk explanatory variables matrix           
            double *Wy,    // Wy = nx1 weight matrix times y-vector
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
			double sig0)   // starting value for sige
{
    // local stuff
    int i, j, iter, pflag;
    double *ys, **xt, **xmat, **xpx, *xpy;
	double **priorv, *priorm, *rhodet, *detval;
	double *bhat, *btmp, *bnorm, **covm, *xb, **xpxix;
    double rho, rmin, rmax, p, epe, chisq;
    double sige, vsqrt, dof, evec, chiv;
	double *e0, *ed, *b0, *bd, *xb0, *xbd, epe0, eped, epe0d;
	BOOL invt;
	   
    // allocate vectors
	xt   = dmatrix(0,k-1,0,n-1);
	xpx  = dmatrix(0,k-1,0,k-1);
	covm = dmatrix(0,k-1,0,k-1);
	xmat = dmatrix(0,n-1,0,k-1);
	priorv = dmatrix(0,k-1,0,k-1);
	xpxix = dmatrix(0,k-1,0,n-1);
	
	ys   = dvector(0,n-1);
	xpy  = dvector(0,k-1);
	priorm = dvector(0,k-1);
	bhat   = dvector(0,k-1);
	btmp   = dvector(0,k-1);
	bnorm  = dvector(0,k-1);
	xb     = dvector(0,n-1);
	rhodet = dvector(0,ngrid-1);
	detval = dvector(0,ngrid-1);

	e0   = dvector(0,n-1);
    ed   = dvector(0,n-1);
    b0   = dvector(0,k-1);
	bd   = dvector(0,k-1);
	xb0  = dvector(0,n-1);
	xbd  = dvector(0,n-1);

	
	// put x into xmat
	// #define X(i,j) x[i + j*n]

	for(i=0; i<n; i++){
		for(j=0; j<k; j++)
			xmat[i][j] = x[i + j*n];
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
 

// do MCMC draws on rho, sige

// ======================================
// start the sampler
// ======================================

for(iter=0; iter<ndraw; iter++){

// apply variance scalars using matmulc
	for(i=0; i<n; i++){
		ys[i] = y[i] - rho*Wy[i];
	}

// ==================================================
// update beta with a multivariate normal draw
	transpose(xmat,n,k,xt);
   
    matmat(xt, k, n, xmat, k, xpx);
    
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
		 mexPrintf("sar_gcc: Inversion error in beta conditional \n");
		 
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

	// form x*bhat
	matvec(xmat, n, k, bhat, xb);

    epe = 0.0;
	for(i=0; i<n; i++){
		evec = y[i] - rho*Wy[i] - xb[i];
		epe = epe + evec*evec;
	}
	chisq = genchi(dof);
    sige = (2*d0 + epe)/chisq;
    

// ==================================================
// update rho using numerical integration

          matmat(xpx, k, k, xt, n, xpxix);
		  matvec(xpxix, k, n, y,  b0);
		  matvec(xpxix, k, n, Wy, bd);	
		  matvec(xmat, n, k, b0, xb0);	
          matvec(xmat, n, k, bd, xbd);

          for(i=0; i<n; i++){
		  e0[i] = y[i]  - xb0[i];
		  ed[i] = Wy[i] - xbd[i];
		  }

		  epe0 = 0.0;
		  eped = 0.0;
		  epe0d = 0.0;
		  for(i=0; i<n; i++){
			  epe0 = epe0 + e0[i]*e0[i];
			  eped = eped + ed[i]*ed[i];
			  epe0d = epe0d + ed[i]*e0[i];
		  }
	
	rho = draw_rho(rhodet,detval,epe0,eped,epe0d,ngrid,n,k,rho);
    

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
	free_dmatrix(xpx,0,k-1,0);
	free_dmatrix(xmat,0,n-1,0);
	free_dmatrix(priorv,0,k-1,0);
	free_dmatrix(covm,0,k-1,0);
	free_dmatrix(xpxix,0,k-1,0);

	free_dvector(xpy,0);
    free_dvector(ys,0);
	free_dvector(bhat,0);
	free_dvector(bnorm,0);
	free_dvector(btmp,0);
	free_dvector(priorm,0);
	free_dvector(xb,0);
	free_dvector(rhodet,0);
	free_dvector(detval,0);

	free_dvector(e0,0);
	free_dvector(ed,0);
	free_dvector(b0,0);
	free_dvector(bd,0);
	free_dvector(xb0,0);
	free_dvector(xbd,0);




} // end of sar_gchom


void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[] )
{
  double *y, *Wy, *x, *ldet, *pdraw, *bdraw, *sdraw, *rdraw, *vmean;
  int  n, k, ngrid, ndraw, nomit;
  double rval, nu, d0, mm, kk;
  double *TI, *TIc, c, T, rho0, sig0;
  long Seed1, Seed2;
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
   if(nrhs == 15){
   flag = 0;
    phrase = phrase2;
  } else if (nrhs == 16){
    flag = 1;
  } else {
    mexErrMsgTxt("sar_gc: 15 or 16 inputs required.");
  }

  if(nlhs != 5) {
    mexErrMsgTxt("sar_gcc: 5 output arguments needed");
  }

    if (flag == 1) {

    // input must be a string
    if ( mxIsChar(prhs[15]) != 1)
      mexErrMsgTxt("sar_gc: seed must be a string.");
    // input must be a row vector
    if (mxGetM(prhs[15])!=1)
      mexErrMsgTxt("sar_gc: seed input must be a row vector.");

    // get the length of the input string
    buflen = (mxGetM(prhs[15]) * mxGetN(prhs[15])) + 1;

    // allocate memory for input string
    phrase = mxCalloc(buflen, sizeof(char));

    // copy the string data from prhs[0] into a C string input_ buf.
    // If the string array contains several rows, they are copied,
    // one column at a time, into one long string array.
    //
    status = mxGetString(prhs[15], phrase, buflen);
    if(status != 0)
      mexWarnMsgTxt("sar_gc: Not enough space. seed string truncated.");
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
    // sar_gcc(y,x,Wy,detval,rval,ndraw,nomit,nu,d0,mm,kk,TI,TIc,p0,sig0);

     y = mxGetPr(prhs[0]);
     x = mxGetPr(prhs[1]);
     n = mxGetM(prhs[1]);
     k = mxGetN(prhs[1]);
	 Wy = mxGetPr(prhs[2]);
	 ldet = mxGetPr(prhs[3]);
	 ngrid = mxGetM(prhs[3]);
	 rval = mxGetScalar(prhs[4]); // rval = 0 is a flag for homoscedastic
	 ndraw = (int) mxGetScalar(prhs[5]);
	 nomit = (int) mxGetScalar(prhs[6]);
	 nu = mxGetScalar(prhs[7]);
	 d0 = mxGetScalar(prhs[8]);
	 mm = mxGetScalar(prhs[9]);
	 kk = mxGetScalar(prhs[10]);
	 TI = mxGetPr(prhs[11]);
	 TIc = mxGetPr(prhs[12]);
	 rho0 = mxGetScalar(prhs[13]);
	 sig0 = mxGetScalar(prhs[14]);
	 
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

    

    if (rval == 0){
    /* Call the homoscedastic  subroutine. */
    sar_gch(pdraw,bdraw,sdraw,y,x,Wy,ldet,n,k,ngrid,ndraw,nomit,nu,d0,TI,TIc,rho0,sig0);
    }else{     /* Call the heteroscedastic  subroutine. */
     sar_gc(pdraw,bdraw,sdraw,rdraw,vmean,y,x,Wy,ldet,n,k,ngrid,rval,ndraw,nomit,nu,d0,mm,kk,TI,TIc,rho0,sig0);
    }   

}


