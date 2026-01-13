
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <malloc.h>
#include <mex.h>
#include "randomlib.h"
#include "matrixjpl.h"


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

	return (rho);


}

// *****************************************************
// main sampler


// far_gc(y,Wy,ldet,rdet,n,ngrid,rval,ndraw,nomit,nu,d0,mm,kk,bdraw,sdraw,rdraw,vmean)

// estimates robust 1st-order spatial autoregressive model using MCMC

void far_gcc(
            double *y,     // y = nx1 lhs vector
            double *Wy,    // Wy = nx1 weight matrix times y-vector
            double *ldet,  // ldet = ngrid x 1 vector with log det values 
            double *rdet,  // rdet = ngrid x 1 vector with associated rho values
            int n,         // n = # of observations
            int ngrid,     // ngrid = # of values in lndet (rows)
            double rval,   // rval = hyperparameter r
            int ndraw,     // ndraw = # of draws
            int nomit,     // nomit = # of burn-in draws to omit
            double nu,     // nu = gamma prior for sige
            double d0,     // d0 = gamma prior for sige
            double mm,     // mm = exp prior for rval
            double kk,     // kk = exp prior for rval
            double *bdraw, // bout = draws for rho (ndraw,1) vector
            double *sdraw, // sout = draws for sige (ndraw,1) vector
            double *rdraw, // rout = draws for rval if mm .ne. 0
            double *vmean  // vmean = mean of vi draws (n,1) vector
            )
{
    // local stuff
    int i, iter, one;
    double *ys, *Wys, *v, rho;
    double rmin, rmax, epe, chisq;
    double sige, vsqrt, dof, evec, chiv;
    double *e0, *ed, epe0, eped, epe0d;
    
    // allocate vectors
    
    ys  = dvector(0,n-1);
    Wys = dvector(0,n-1);
    v   = dvector(0,n-1);
	e0   = dvector(0,n-1);
    ed   = dvector(0,n-1);
  
	
// initializations

    dof = ((double) n + nu);
    evec = 0.0;
    sige = 1.0;
    one = 1;
	for(i=0; i<n; i++){
		v[i] = 1.0;
	}
	
    rmin = rdet[0];
    rmax = rdet[ngrid-1];
    
    rho = 0.5;


// do MCMC draws on rho, sige, V

// ======================================
// start the sampler
// ======================================

for(iter=0; iter<ndraw; iter++){

// apply variance scalars using matmulc
	for(i=0; i<n; i++){
		vsqrt = sqrt(v[i]);
		ys[i] = y[i]/vsqrt;
		Wys[i] = Wy[i]/vsqrt;
	}
		
// ==================================================
// update sigma with a chi-squared draw
    epe = 0.0;
	for(i=0; i<n; i++){
		evec = ys[i] - rho*Wys[i];
		epe = epe + evec*evec;
	}
	chisq = genchi(dof);
    sige = (d0 + epe)/chisq;
    
// ==================================================
// update vi with a chi-squared draw
	for(i=0; i<n; i++){
    evec = y[i] - rho*Wy[i];
	chiv = genchi(rval+1.0);
	v[i] = ((evec*evec/sige) + rval)/chiv;
	}

// ==================================================
// update rho using numerical integration


          for(i=0; i<n; i++){
		  e0[i] = ys[i];
		  ed[i] = Wys[i];
		  }

		  epe0 = 0.0;
		  eped = 0.0;
		  epe0d = 0.0;
		  for(i=0; i<n; i++){
			  epe0 = epe0 + e0[i]*e0[i];
			  eped = eped + ed[i]*ed[i];
			  epe0d = epe0d + ed[i]*e0[i];
		  }
	
	rho = draw_rho(rdet,ldet,epe0,eped,epe0d,ngrid,n,one,rho);
    
//==================================================
// update rval if mm .ne. 0
    if (mm != 0.0){
    rval = gengam(mm,kk);
    }
    
     
// ==================================================
// save the draws
   *(sdraw+iter) = sige;
   *(bdraw+iter) = rho;
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
    free_dvector(ys,0);
	free_dvector(Wys,0);
    free_dvector(v,0);
	free_dvector(e0,0);
	free_dvector(ed,0);

    

} // end of far_gc

// **************************************************************

void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[] )
{
  double *y, *Wy, *ldet, *rdet, *bdraw, *sdraw, *rdraw, *vmean;
  int  n, ngrid, ndraw, nomit;
  double c, T, rval, nu, d0, mm, kk;
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


//  [rout,sout,rdraw,vmean] = ...
// far_gcc(y,Wy,detval(:,2),detval(:,1),n,length(detval),rval,ndraw,nomit,nu,d0,mm,kk,prho,pvar);

   if(nrhs == 13){
   flag = 0;
    phrase = phrase2;
  } else if (nrhs == 14){
    flag = 1;
  } else {
    mexErrMsgTxt("far_gc: 13 or 14 inputs required.");
  }

  if(nlhs != 4) {
    mexErrMsgTxt("far_gc: 4 output arguments needed");
  }
  
    if (flag == 1) {

    // input must be a string
    if ( mxIsChar(prhs[13]) != 1)
      mexErrMsgTxt("far_gc: seed must be a string.");
    // input must be a row vector
    if (mxGetM(prhs[13])!=1)
      mexErrMsgTxt("far_gc: seed input must be a row vector.");

    // get the length of the input string
    buflen = (mxGetM(prhs[13]) * mxGetN(prhs[13])) + 1;

    // allocate memory for input string
    phrase = mxCalloc(buflen, sizeof(char));

    // copy the string data from prhs[0] into a C string input_ buf.
    // If the string array contains several rows, they are copied,
    // one column at a time, into one long string array.
    //
    status = mxGetString(prhs[13], phrase, buflen);
    if(status != 0)
      mexWarnMsgTxt("far_gc: Not enough space. seed string truncated.");
    }
    
    // allow the user to set a seed or rely on clock-based seed
   if (flag == 0) {
    setSeedTimeCore(phrase);
   } else {
	phrtsd(phrase,&Seed1,&Seed2);
    setall(Seed1,Seed2);
   }


  	//  parse input arguments
     y     = mxGetPr(prhs[0]);
	 Wy    = mxGetPr(prhs[1]);
	 ldet  = mxGetPr(prhs[2]);
	 rdet  = mxGetPr(prhs[3]);
	 n     = (int) mxGetScalar(prhs[4]);
	 ngrid = (int) mxGetScalar(prhs[5]);
	 rval  = mxGetScalar(prhs[6]);
	 ndraw = (int) mxGetScalar(prhs[7]);
	 nomit = (int) mxGetScalar(prhs[8]);
	 nu = mxGetScalar(prhs[9]);
	 d0 = mxGetScalar(prhs[10]);
	 mm = mxGetScalar(prhs[11]);
	 kk = mxGetScalar(prhs[12]);

	// no need for error checking on inputs
	// since this was done in the matlab function

    /* Create matrices for the return arguments */
	 //bdraw,sdraw,rdraw,vmean
    plhs[0] = mxCreateDoubleMatrix(ndraw,1, mxREAL);
	plhs[1] = mxCreateDoubleMatrix(ndraw,1, mxREAL);
	plhs[2] = mxCreateDoubleMatrix(ndraw,1, mxREAL);
	plhs[3] = mxCreateDoubleMatrix(n,1, mxREAL);

    bdraw = mxGetPr(plhs[0]);
	sdraw = mxGetPr(plhs[1]);
	rdraw = mxGetPr(plhs[2]);
	vmean = mxGetPr(plhs[3]);

    /* Call the  subroutine. */
    far_gcc(y,Wy,ldet,rdet,n,ngrid,rval,ndraw,nomit,nu,d0,mm,kk,bdraw,sdraw,rdraw,vmean);


}


