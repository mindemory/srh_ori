library(dplyr)
library(permuco)
library(tidyr)

data <- read.csv('/d/DATD/datd/MD_TMS_EEG/analysis/meta_analysis/calib_all5_filtered.csv')
data_ave <- summarize(group_by(data, subjID, TMS_time, instimVF, hemistimulated),
                                mean_ierr=mean(ierr),
                                mean_ferr=mean(ferr),
                                mean_isaccrt=mean(isacc_rt),
                                mean_iradial=mean(iradial))
# Setting factors
data_ave$TMS_time.f <- as.factor(data_ave$TMS_time)
contrasts(data_ave$TMS_time.f) <- contr.treatment
contrasts(data_ave$TMS_time.f)

data_ave$instimVF.f <- as.factor(data_ave$instimVF)
contrasts(data_ave$instimVF.f) <- contr.treatment
contrasts(data_ave$instimVF.f)

data_ave$hemistimulated.f <- as.factor(data_ave$hemistimulated)
contrasts(data_ave$hemistimulated.f) <- contr.treatment
contrasts(data_ave$hemistimulated.f)

aov.ierr.perm <- aovperm(mean_ierr ~ TMS_time.f * instimVF.f +
                        Error(subjID/(TMS_time.f * instimVF.f)),
                        data=data_ave, np=10000)
summary(aov.ierr.perm)

# aov.ferr.perm <- aovperm(mean_ferr ~ TMS_time.f * instimVF.f +
#                         Error(subjID/(TMS_time.f * instimVF.f)),
#                         data=data_ave, np=10000)
# summary(aov.ferr.perm)

# aov.irt.perm <- aovperm(mean_isaccrt ~ TMS_time.f * instimVF.f +
#                         Error(subjID/(TMS_time.f * instimVF.f)),
#                         data=data_ave, np=10000)
# summary(aov.irt.perm)

# aov.iradial.perm <- aovperm(mean_iradial ~ TMS_time.f * instimVF.f +
#                         Error(subjID/(TMS_time.f * instimVF.f)),
#                         data=data_ave, np=10000)
# summary(aov.iradial.perm)