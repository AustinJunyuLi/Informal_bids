clear
set mem 500m
set matsize 800
set more 1

use Cohen_and_Einav_data

log using tables1_2_3.log, replace

* construct variables for Table 1

gen XindAge = individualAge 
gen XindAge2 = individualAge^2
ren XindAge2 nXindAge2
gen XindFemale = individualGender
gen XindSingle     = individualMaritalStatus==1 /* to be omitted later */
gen XindMarried   = individualMaritalStatus==2
gen XindDivorced = individualMaritalStatus==3
gen XindWidower = individualMaritalStatus==4
gen XindMaritalNores = individualMaritalStatus==0
gen XindElementary = individualEducation==1
gen XindHighSchool = individualEducation==2 /* to be omitted later */
gen XindTechnical = individualEducation==3
gen XindAcademic = individualEducation==4
gen XindEduNores = individualEducation==0
gen XindEmigrant = (individualEmigrationYear>0)

gen XcarValue = carValue
gen XcarAge = carAge
gen XcarCommercial = carCommercial
gen XcarEngine = carEngineSize

gen XdrLicense = drivingLicenseYears
gen XdrLicense2 = drivingLicenseYears^2
ren XdrLicense2 nXdrLicense2
gen XdrGoodDriver = drivingGoodDriver
gen XdrAnyDriver = drivingAnyDriverClause
gen XdrSecondary = drivingSecondaryCar
gen XdrBusiness = drivingBusinessUse
gen XdrMileage = drivingEstimatedMileage /* to be omitted for most specs */
gen XdrHistLength = drivingHistoryLength
gen XdrHistClaims = drivingClaimHistory

gen Xyoung = youngAge>0
gen XyoungMale = youngGender==1 /* to be omitted later */
gen XyoungFemale = youngGender==2
gen XyoungAge1719 = youngAge==1 /* to be omitted later */
gen XyoungAge1921 = youngAge==2
gen XyoungAge2124 = youngAge==3
gen XyoungAge24p   = youngAge==4
gen XyoungExp1m    = youngExperience==1 /* to be omitted later */
gen XyoungExp13    = youngExperience==2
gen XyoungExp3p    = youngExperience==3

gen Xyr1 = policyCompanyYear==1 /* to be omitted later */
gen Xyr2 = policyCompanyYear==2
gen Xyr3 = policyCompanyYear==3
gen Xyr4 = policyCompanyYear==4
gen Xyr5 = policyCompanyYear==5

*************
/* Table 1 */
*************
sum X*

ren XdrMileage nXdrMileage
ren XindSingle nXindSingle
ren XindHighSchool nXindHighSchool
ren XyoungMale nXyoungMale
ren XyoungAge1719 nXyoungAge1719
ren XyoungExp1m nXyoungExp1m
ren Xyr1 nXyr1

ren nXindAge2 XindAge2
ren nXdrLicense2 XdrLicense2
replace XcarEngine = log(carEngineSize)
replace XcarValue = log(carValue)

* construct variables for Table 2

gen YmenuDedL = menuDeductible1
gen YmenuDedR = menuDeductible2
gen YmenuDedH = menuDeductible3
gen YmenuDedVH = menuDeductible4
gen YmenuPrL = menuPremium1
gen YmenuPrR = menuPremium2
gen YmenuPrH = menuPremium3
gen YmenuPrVH = menuPremium4
gen YmenuRatio = (menuPremium1-menuPremium2)/(menuDeductible2-menuDeductible1)
gen YchoiceL = outcomeDeductibleChoice==1
gen YchoiceR = outcomeDeductibleChoice==2
gen YchoiceH = outcomeDeductibleChoice==3
gen YchoiceVH = outcomeDeductibleChoice==4
gen YstatusTrunc = policyStatus==1
gen YstatusCancl = policyStatus==2
gen YstatusExpir = policyStatus==3
gen Yduration = policyDuration
gen Yclaims = outcomeClaimsNumber
gen YclaimsIfL = outcomeClaimsNumber if YchoiceL==1
gen YclaimsIfR = outcomeClaimsNumber if YchoiceR==1
gen YclaimsIfH = outcomeClaimsNumber if YchoiceH==1
gen YclaimsIfVH = outcomeClaimsNumber if YchoiceVH==1

gen wYclaimsAdj = Yclaims/Yduration
gen wYclaimsAdjL = wYclaimsAdj if YchoiceL==1
gen wYclaimsAdjR = wYclaimsAdj if YchoiceR==1
gen wYclaimsAdjH = wYclaimsAdj if YchoiceH==1
gen wYclaimsAdjVH = wYclaimsAdj if YchoiceVH==1

**************
/* Table 2A */
**************
sum Y*
sum wY* [aweight=Yduration]

**************
/* Table 2B */
**************
tab outcomeClaimsNumber outcomeDeductibleChoice if policyDuration>0.9

******************************
/* Table 3 - Poisson (Coef) */
******************************
poisson outcomeClaimsNumber X*, exposure(policyDuration)

*****************************
/* Table 3 - Poisson (IRR) */
*****************************
poisson outcomeClaimsNumber X*, exposure(policyDuration) irr

predict LambdaHat, ir
gen menuDbar  = (menuDeductible1+menuDeductible2)/2
gen choiceThres = (YmenuRatio/LambdaHat - 1) / menuDbar
gen LogchoiceThres = -log(choiceThres)

***********************************************************
/* Table 3 - Probit, column (2) - coefficinets BEFORE renormalization */
***********************************************************
probit YchoiceL LogchoiceThres X* /* get the coefficients, not dp/dx, and use to renormalize */
predict lrHat, xb
replace lrHat = lrHat/.0947093 /* the coefficient on the threshold */ 

****************************************
/* Table 3 - Probit, column (2) - IRR */
****************************************
dprobit YchoiceL LogchoiceThres X*

****************************************
/* Table 3 - Probit, column (3) - IRR */
****************************************
dprobit YchoiceL X*
log close









