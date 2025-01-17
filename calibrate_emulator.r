# Required packages
library(nloptr)
library(expm)
library(FKF)

# Constants
MAX_EVAL <- 1e+05
XTOL_REL <- 1e-5

#' Log transform parameters
#'
#' @param parameters List of parameters
#' @return Numeric vector of log-transformed parameters
LogTransform <- function(parameters) {
    # use positive parameter domain, except for epsilon
    parameters$epsilon <- exp(parameters$epsilon)
    p <- log(unlist(parameters))
    return(p)
}

#' Exponential transform parameters
#'
#' @param p Numeric vector of parameters
#' @return List of exp-transformed parameters
ExpTransform <- function(p) {
    p <- exp(unname(p))
    parameters <- list(
        gamma = p[1],
        chi = p[2:3],
        kappa = p[4:5],
        epsilon = log(p[6]),
        sigma = p[7:9],
        Fbar = p[10]
    )
    return(parameters)
}

#' Build A, B, and V matrices
#'
#' @param gamma Numeric
#' @param chi Numeric vector
#' @param kappa Numeric vector
#' @param epsilon Numeric
#' @param sigma Numeric vector
#' @return List containing A, B, and V matrices
BuildABV <- function(gamma, chi, kappa, epsilon, sigma) {
    A <- rbind(
        c(-gamma, 0, 0),
        c(1/chi[1], -(kappa[1] + epsilon + kappa[2])/chi[1], (kappa[2] + epsilon)/chi[1]),
        c(0, kappa[2]/chi[2], -kappa[2]/chi[2])
    )
    m <- nrow(A)
    B <- matrix(0, m)
    B[1] <- gamma
    V <- matrix(0, m, m)
    V[1,1] <- sigma[1]^2
    V[2,2] <- (sigma[2]/chi[1])^2
    return(list(A=A, B=B, V=V))
}

#' Build Ad, Bd, and Vd matrices
#'
#' @param A Matrix
#' @param B Matrix
#' @param V Matrix
#' @return List containing Ad, Bd, and Vd matrices
BuildAdBdVd <- function(A, B, V) {
    m <- nrow(A)
    Ad <- expm::expm(A)
    Bd <- solve(A, (Ad - diag(m)) %*% B)
    D <- rbind(
        cbind(-A, V),
        cbind(matrix(0, m, m), t(A))
    )
    F <- expm::expm(D)
    Vd <- t(F[(m + 1):(2*m), (m + 1):(2*m)]) %*% F[1:m, (m + 1):(2*m)]
    return(list(Ad=Ad, Bd=Bd, Vd=Vd))
}

#' Build P0 matrix
#'
#' @param Ad Matrix
#' @param Vd Matrix
#' @return List containing P0 matrix
BuildP0 <- function(Ad, Vd) {
    m <- nrow(Ad)
    vecP0 <- solve(diag(m^2) - kronecker(Ad, Ad), as.vector(Vd))
    P0 <- matrix(vecP0, m)
    return(list(P0=P0))
}

#' Build Cd and Wd matrices
#'
#' @param kappa Numeric vector
#' @param epsilon Numeric
#' @return List containing Cd and Wd matrices
BuildCdWd <- function(kappa, epsilon) {
    Cd <- rbind(
        c(0, 1, 0),
        c(1, -kappa[1] - epsilon, epsilon)
    )
    Wd <- diag(1e-12, nrow(Cd))
    return(list(Cd=Cd, Wd=Wd))
}

#' Build all matrices
#'
#' @param gamma Numeric
#' @param chi Numeric vector
#' @param kappa Numeric vector
#' @param epsilon Numeric
#' @param sigma Numeric vector
#' @return List containing all matrices
BuildMatrices <- function(gamma, chi, kappa, epsilon, sigma) {
    ABV <- BuildABV(gamma, chi, kappa, epsilon, sigma)
    AdBdVd <- with(ABV, BuildAdBdVd(A, B, V))
    P0 <- with(AdBdVd, BuildP0(Ad, Vd))
    CdWd <- BuildCdWd(kappa, epsilon)
    return(c(AdBdVd, P0, CdWd))
}

#' Perform Kalman filtering
#'
#' @param Ad Matrix
#' @param Bd Matrix
#' @param Vd Matrix
#' @param P0 Matrix
#' @param Cd Matrix
#' @param Wd Matrix
#' @param Fbar Numeric
#' @param dataset Matrix
#' @return Kalman filter result
KalmanFilter <- function(Ad, Bd, Vd, P0, Cd, Wd, Fbar, dataset) {
    m <- nrow(Ad)
    d <- nrow(Cd)

    a0 <- c(Fbar, rep(0, m-1))
    a0 <- as.vector(Ad %*% a0 + Bd*Fbar)
    Tt <- array(Ad, c(m, m, 1))
    dt <- Bd*Fbar
    HHt <- array(Vd, c(m, m, 1))

    yt <- dataset
    Zt <- array(Cd, c(d, m, 1))
    ct <- matrix(0, d, 1)
    GGt <- array(Wd, c(d, d, 1))

    tryCatch({
        kf <- FKF::fkf(a0, P0, dt, ct, Tt, Zt, HHt, GGt, yt)
        return(kf)
    }, error = function(e) {
        stop(paste("Error in Kalman filter:", e$message))
    })
}

#' Calculate negative log-likelihood for Kalman filter
#'
#' @param p Numeric vector of parameters
#' @param dataset Matrix
#' @return Negative log-likelihood value
KalmanNegLogLik <- function(p, dataset) {
    parameters <- ExpTransform(p)
    Matrices <- with(parameters, BuildMatrices(gamma, chi, kappa, epsilon, sigma))
    kf <- with(c(parameters, Matrices), KalmanFilter(Ad, Bd, Vd, P0, Cd, Wd, Fbar, dataset))
    val <- -kf$logLik
    print(val)
    return(val)
}

#' Fit Kalman filter
#'
#' @param parameters List of initial parameters
#' @param T1 Numeric vector
#' @param R Numeric vector
#' @param maxeval Integer
#' @return List containing optimization results
FitKalman <- function(parameters, T1, R, maxeval=MAX_EVAL) {
    p <- LogTransform(parameters)
    dataset <- rbind(T1, R)

    res <- nloptr::bobyqa(
        x0 = p,
        fn = KalmanNegLogLik,
        dataset = dataset,
        control = list(maxeval=maxeval, xtol_rel=XTOL_REL)
    )

    obj <- res$value
    itr <- res$iter
    mle <- res$par

    parameters <- ExpTransform(mle)
    return(list(
        obj = obj,
        itr = itr,
        mle = mle,
        parameters = parameters)
    )
}

#' Load and process climate data
#'
#' @param model Character
#' @param experiment Character
#' @param variant Character
#' @return List containing processed data
load_climate_data <- function(model, experiment, variant) {
    data_dir <- "./data_processed"

    # Load data
    data_experiment <- read.csv(file.path(data_dir, sprintf("tas_%s_%s_%s.csv", model, experiment, variant)))
    data_piControl <- read.csv(file.path(data_dir, sprintf("tas_%s_piControl_%s.csv", model, variant)))
    rsdt <- read.csv(file.path(data_dir, sprintf("rsdt_%s_%s_%s.csv", model, experiment, variant)))$rsdt
    rsut <- read.csv(file.path(data_dir, sprintf("rsut_%s_%s_%s.csv", model, experiment, variant)))$rsut
    rlut <- read.csv(file.path(data_dir, sprintf("rlut_%s_%s_%s.csv", model, experiment, variant)))$rlut

    # Process data
    tas <- data_experiment$tas
    tas_control_mean <- mean(data_piControl$tas)
    T1 <- tas - tas_control_mean
    R <- rsdt - rlut - rsut

    return(list(T1 = T1, R = R))
}

#' Main execution function
#'
#' @param model Character
#' @param experiment Character
#' @param variant Character
#' @return None
main <- function(model, experiment, variant) {
    tryCatch({
        # Load and process data
        data <- load_climate_data(model, experiment, variant)

        # Initial parameters
        parameters <- list(
            gamma = 1,
            chi = c(5, 100),
            kappa = c(1, 1),
            epsilon = 0.1,
            sigma = c(0.5, 0.5, 0.5),
            Fbar = 5
        )

        # Fit Kalman filter
        result <- with(data, {
            FitKalman(parameters=parameters, T1=T1, R=R)
        })

        parameters <- unlist(result$parameters)

        # Print results
        cat("Estimated parameters:\n")
        print(parameters)
        cat("\nObjective:", result$obj, "\n")
        cat("Iterations:", result$itr, "\n")

        # Save results
        write.csv(parameters,
                  sprintf('./output/parameter_%s_%s_%s.csv', model, experiment, variant))
        write.csv(data.frame(T1=data$T1, R=data$R), 
                  sprintf("./output/df_%s_%s_%s.csv", model, experiment, variant))

    }, error = function(e) {
        cat("Error in main execution:", e$message, "\n")
    })
}

args <- commandArgs(trailingOnly=TRUE)
if (length(args) == 0) {
    model <- "MIROC6"
} else {
    model <- args[1]
}

experiment <- "abrupt-4xCO2"
variant <- "r1i1p1f1"

main(model, experiment, variant)
