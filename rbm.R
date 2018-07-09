source('utils')

# Função de ativação
funcao.ativacao <- function(v) {
    return(1 / (1 + exp(-v)))
}

# Função para fazer a amostragem
sample.bernoulli <- function(p) {
    retorno <- 0

    if(runif(min=0, max=1, 1) <= p){
        retorno <- 1
    }

    return(retorno)
}

# Função para construir a arquitetura
arquitetura <- function(num.visiveis, num.escondidos, n, funcao.ativacao) {
    arq <- list()
    # Parametros da rede
    arq$num.visiveis <- num.visiveis
    arq$num.escondidos <- num.escondidos
    arq$funcao.ativacao <- funcao.ativacao
    arq$n <- n

    # 2 neuronios na camada escondida
    #
    #      Ent1    Ent2    Bias
    # 1    w11     w12     w13
    # 2    w21     w22     w23

    # Pesos conectando camada visivel e escondida
    num.pesos.visivel.escondida <- num.visiveis * num.escondidos
    arq$visivel.escondida <- matrix(runif(min=-0.5, max=0.5, num.pesos.visivel.escondida), nrow = num.escondidos, ncol = num.visiveis)

    # Pesos conectando bias e camada escondida
    arq$bias.escondida <- runif(min=-0.5, max=0.5, num.escondidos)
    # Pesos conectando bias e camada visivel
    arq$bias.visivel <- runif(min=-0.5, max=0.5, num.visiveis)

    return(arq)
}

# Função para a execução do algoritmo contrastive divergence
contrastive.divergence <- function(exemplos, arq, K, epocas) {
    
    cat("\n\nContrastive Divergence... \n")

    # Para cada epoca
    for(i in 1:epocas){
        # Para cada exemplo
        for(j in 1:length(exemplos)){
            x <- exemplos[[j]]

            bernoulli.p.x.h <- x 

            # Para cada K (Gibbs Sampling)
            for(k in 1:K) {
                # p(h|x)
                v.h.x <- cbind(arq$visivel.escondida, arq$bias.escondida) %*% c(bernoulli.p.x.h,1)
                p.h.x <- as.numeric(arq$funcao.ativacao(v.h.x))
                bernoulli.p.h.x <- unlist(lapply(p.h.x, sample.bernoulli))

                # p(x|h) (reconstrucao)
                v.x.h <- t(c(bernoulli.p.h.x, 1)) %*% rbind(arq$visivel.escondida, arq$bias.visivel)
                p.x.h <- as.numeric(arq$funcao.ativacao(v.x.h))
                bernoulli.p.x.h <- unlist(lapply(p.x.h, sample.bernoulli))
            }

            # p(h|~x) -> h(~x)
            v.h.x.reconst <- cbind(arq$visivel.escondida, arq$bias.escondida) %*% c(bernoulli.p.x.h,1)
            p.h.x.reconst <- as.numeric(arq$funcao.ativacao(v.h.x.reconst))

            # Atualização dos pesos
            deltaW <- p.h.x %*% t(x) - p.h.x.reconst %*% t(bernoulli.p.x.h)
            arq$visivel.escondida <- arq$visivel.escondida + arq$n * deltaW

            arq$bias.escondida <- arq$bias.escondida + arq$n * (p.h.x - p.h.x.reconst)

            arq$visivel <- arq$bias.visivel + arq$n * (x - bernoulli.p.x.h)
        }
    }

    return(arq)

}

# Reconstrução
reconstrucao <- function(modelo, padrao) {
    #p (h|x)
    v.h.x <- cbind(modelo$visivel.escondida, arq$bias.escondida) %*% c(padrao,1)
    p.h.x <- as.numeric(modelo$funcao.ativacao(v.h.x))
    bernoulli.p.h.x <- unlist(lapply(p.h.x, sample.bernoulli))

    # p(x|h)
    v.x.h <- t(c(bernoulli.p.h.x, 1)) %*% rbind(modelo$visivel.escondida, modelo$bias.visivel)
    p.x.h <- as.numeric(modelo$funcao.ativacao(v.x.h))
    bernoulli.p.x.h <- unlist(lapply(p.x.h, sample.bernoulli))

    return(bernoulli.p.x.h)
}

# carrega.digitos(digitos=c(0,1,2,3,4,5,6,7,8,9), 100, 1, 0.5)

# arq <- arquitetura(length(padroes[[1]]), length(padroes[[1]])-1, 0.2, funcao.ativacao)

# modelo <- contrastive.divergence(padroes, arq, 1, 10)

# testa.digitos(modelo)

