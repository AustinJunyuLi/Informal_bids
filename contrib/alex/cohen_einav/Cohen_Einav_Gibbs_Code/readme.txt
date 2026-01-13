
Data and Estimation programs for "Estimating Risk Preferences from Deductible Choice" by Alma Cohen and Liran Einav
-------------------------------------------------------------------------------------------------------------------

This is a readme file that explains what files are included in the "estimation programs" directory and how the data necessary to run the programs could be obtained. 


A researcher seeking the data will need to sign an agreement with Alma Cohen committing the researcher to use the data solely for the purpose of performing replication and robustness tests of the tables in the above paper, not to use the data in any other way, and not to transfer the data to any other party. A researcher seeking to enter such an agreement and get the data to perform replications and robustensss tests should contact Alma Cohen at alcohen@post.harvard.edu. 


Once the data is obtained, the programs can be used to generate Tables 1-9 of the paper as follows: 

The first set of files generate the summary statistics and reduced form results reported in Tables 1-3:

- Cohen_and_Einav_data.dta - this is the full data set, in Stata 9 format (THIS FILE SHOULD BE REQUESTED SEPARATELY).
- tables1_2_3.do - this is a stata do file that generates Tables 1, 2, and 3 in the paper. It uses the Cohen_and_Einav_data.dta data file.
- tables1_2_3.log is the output file from running the corresponding do file, and is identical to Tables 1, 2, and 3 in the paper.

The second set of files run the Gibbs smapler for the estimation of the benchmark model, and reproduce the results reported in Table 4:

- Cohen_and_Einav_data.mat - this is the full data set, in Matlab 7.1 format. It is identical to the Stata file above, with the only difference is that missing values are replaced with -999 (THIS FILE SHOULD BE REQUESTED SEPARATELY).
- main_program.m is the main program: it loads and organizes the data, and then calls the other programs.
- gibbs_benchmark.m is the file where the Gibbs sampler algorithm is implemented. (note: Gibbs samplers for alternative specifications reported in the paper are available upon request; they are very similar).
- myinv.m is a simple function that efficiently inverts a two-by-two matrix.
- output.m is a function that generates the output.
- results_benhcmark.log is the output file, which can be reproduced by placing all the above files in the same directory and running the main_program file. The output is identical to Table 4 in the paper.

The definitions of most variables should be self explanatory from their names. Full variable definitions are available in Appendix B of the paper. In case of doubt, simply follow the stata do file above that generates Tables 1 and 2, and compare the generated tables with the tables in the paper. The variables appear in the same order. 

If there are any problems or questions about the use of the estimation programs, please email leinav@stanford.edu.

Thanks for your interest.

Good luck!
