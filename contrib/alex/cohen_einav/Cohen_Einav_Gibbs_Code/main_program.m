clear;

load Cohen_and_Einav_data.mat;

% resetting the random number generators
rand('state',1);
randn('state',1);

% Naming the variables
policyCompanyYear = cleaned_data(:,1);
policyDate = cleaned_data(:,2);
policyDay = cleaned_data(:,3);
policyDuration = cleaned_data(:,4);
policyIndividual = cleaned_data(:,5);
policyMonth = cleaned_data(:,6);
policyStatus = cleaned_data(:,7);
policyYear = cleaned_data(:,8);
youngAge = cleaned_data(:,9);
youngExperience = cleaned_data(:,10);
youngGender = cleaned_data(:,11);
drivingAnyDriverClause = cleaned_data(:,12);
drivingBusinessUse = cleaned_data(:,13);
drivingClaimHistory = cleaned_data(:,14);
drivingEstimatedMileage = cleaned_data(:,15);
drivingGoodDriver = cleaned_data(:,16);
drivingHistoryLength = cleaned_data(:,17);
drivingLicenseYears = cleaned_data(:,18);
drivingSecondaryCar = cleaned_data(:,19);
indRenewals = cleaned_data(:,20);
individualAge = cleaned_data(:,21);
individualCompanyEmployee = cleaned_data(:,22);
individualEducation = cleaned_data(:,23);
individualEmigrationYear = cleaned_data(:,24);
individualGender = cleaned_data(:,25);
individualMaritalStatus = cleaned_data(:,26);
individualReferral = cleaned_data(:,27);
carAge = cleaned_data(:,28);
carCommercial = cleaned_data(:,29);
carEngineSize = cleaned_data(:,30);
carValue = cleaned_data(:,31);
menuDeductible1 = cleaned_data(:,32);
menuDeductible2 = cleaned_data(:,33);
menuDeductible3 = cleaned_data(:,34);
menuDeductible4 = cleaned_data(:,35);
menuPremium1 = cleaned_data(:,36);
menuPremium2 = cleaned_data(:,37);
menuPremium3 = cleaned_data(:,38);
menuPremium4 = cleaned_data(:,39);
censusIncomeIndex = cleaned_data(:,40);
censusIncomeMatchedHH = cleaned_data(:,41);
censusIncomeMatchedInd = cleaned_data(:,42);
censusIncomeMeanHH = cleaned_data(:,43);
censusIncomeMeanInd = cleaned_data(:,44);
censusPopulation = cleaned_data(:,45);
censusTractCode = cleaned_data(:,46);
otherCoverageRadio = cleaned_data(:,47);
otherCoverageTowing = cleaned_data(:,48);
otherCoverageWindshield = cleaned_data(:,49);
outcomeClaimsNumber = cleaned_data(:,50);
outcomeDeductibleChoice = cleaned_data(:,51);

clear cleaned_data;

% Constructing all potential regressors

XindAge = individualAge; 
XindAge2 = individualAge.^2;
XindFemale = individualGender;
% XindSingle = individualMaritalStatus==1 /* omitted var */
XindMarried   = individualMaritalStatus==2;
XindDivorced = individualMaritalStatus==3;
XindWidower = individualMaritalStatus==4;
XindMaritalNores = individualMaritalStatus==0;
XindElementary = individualEducation==1;
% XindHighSchool = individualEducation==2 /* omitted var */
XindTechnical = individualEducation==3;
XindAcademic = individualEducation==4;
XindEduNores = individualEducation==0;
XindEmigrant = (individualEmigrationYear>0);

XcarValue = log(carValue);
XcarAge = carAge;
XcarCommercial = carCommercial;
XcarEngine = log(carEngineSize);

XdrLicense = drivingLicenseYears;
XdrLicense2 = drivingLicenseYears.^2;
XdrGoodDriver = drivingGoodDriver;
XdrAnyDriver = drivingAnyDriverClause;
XdrSecondary = drivingSecondaryCar;
XdrBusiness = drivingBusinessUse;
XdrHistLength = drivingHistoryLength;
XdrHistClaims = drivingClaimHistory;
% XdrMileage = drivingEstimatedMileage; % to be omitted for most specs

Xyoung = youngAge>0;
% XyoungMale = youngGender==1 /* omitted var */
XyoungFemale = youngGender==2;
% XyoungAge1719 = youngAge==1 /* omitted var */
XyoungAge1921 = youngAge==2;
XyoungAge2124 = youngAge==3;
XyoungAge24p   = youngAge==4;
% XyoungExp1m    = youngExperience==1 /* omitted var */
XyoungExp13    = youngExperience==2;
XyoungExp3p    = youngExperience==3;

% Xyr1 = policyCompanyYear==1 /* omitted var */
Xyr2 = policyCompanyYear==2;
Xyr3 = policyCompanyYear==3;
Xyr4 = policyCompanyYear==4;
Xyr5 = policyCompanyYear==5;

myX = [XindAge XindAge2 XindFemale XindMarried XindDivorced XindWidower XindMaritalNores XindElementary XindTechnical XindAcademic ...
    XindEduNores XindEmigrant XcarValue XcarAge XcarCommercial XcarEngine XdrLicense XdrLicense2 XdrGoodDriver XdrAnyDriver ...
    XdrSecondary XdrBusiness XdrHistLength XdrHistClaims Xyoung XyoungFemale XyoungAge1921 XyoungAge2124 XyoungAge24p ...
    XyoungExp13 XyoungExp3p Xyr2 Xyr3 Xyr4 Xyr5];

clear X*;

k = size(myX,2) + 1; % +1 for the constant, which comes in the end
Xl_ind = ones(k,1);
Xr_ind = [ones(k-11,1); zeros(6,1); ones(5,1)]; % k by 1 indicators vector with 1 for X's in te r equation, +1 for the constant

choice   = outcomeDeductibleChoice;
ratio    = (menuPremium1-menuPremium2)./(menuDeductible2-menuDeductible1);
dbar     = (menuDeductible1+menuDeductible2)/2;
claims   = outcomeClaimsNumber;
exposure = policyDuration;

SampleInd = logical(ones(length(carValue),1)); % sample definition

clear policy* young* driving* individual* car* menu* census* other* outcome*;

% main program
gibbs_benchmark;

% output
diary results_benchmark.log;
output;
diary off;



