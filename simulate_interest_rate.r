# load data
spec <- "y10"
filename <- paste0("data_raw/Bauer2023/uc_estimates_", spec, ".RData")
load(filename)

# show all data object loaded
ls.str()

sim_uc_r <- function(rstar0, sigu, sigv, phi, LB=0, M=50000, N=400) {
    ## simulate interestrate based on the extimated UC model
    stopifnot(all.equal(length(rstar0), length(sigu), length(sigv), length(phi)))
    stopifnot(length(rstar0) == M)
    rstar <- matrix(0, M, N)
    rtilde <- matrix(0, M, N)
    rstar[,1] <- rstar0
    rtilde[,1] <- 0   # no need to match current short rate
    for (n in 2:N) {
        rstar[,n] <- rstar[,n-1] + rnorm(M, 0, sigu)
        rtilde[,n] <- phi*rtilde[,n-1] + rnorm(M, 0, sigv)
    }
    r <- rstar + rtilde
    r[r < LB] <- LB # non-negativity constraint
    return(r)
}

LB <- 0
year1 <- 1990
year2 <- 2019

t1 <- which(data$year == year1)
t2 <- which(data$year == year2)
rstar_1 <- X[,1,t1]
rstar_2 <- X[,1,t2]
rstar_1_mean <- mean(rstar_1)
rstar_2_mean <- mean(rstar_2)

r1 <- sim_uc_r(rstar_1, sqrt(sigeta2), sqrt(sigv2), phi, LB=LB)
r2 <- sim_uc_r(rstar_2, sqrt(sigeta2), sqrt(sigv2), phi, LB=LB)

write.csv(r1, sprintf('./data_processed/interest_rate/interest_rate_%s_%s.csv', spec, year1))
write.csv(r2, sprintf('./data_processed/interest_rate/interest_rate_%s_%s.csv', spec, year2))
