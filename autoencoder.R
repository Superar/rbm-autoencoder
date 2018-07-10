source('utils.R')

# Funcao de ativacao
funcao_ativacao <- function(v) {
  y <- 1 / (1 + exp(-v))
  return(y)
}

# Derivada da funcao de ativacao
der_funcao_ativacao <- function(y) {
  derivada <- y * (1 - y)
  return(derivada)
}

# Arquitetura do Autoencoder
arquitetura <- function(num_entrada, num_escondida,
                        funcao_ativacao, der_funcao_ativacao) {
  arq <- list()

  # Parametros da rede
  arq$num_entrada <- num_entrada
  arq$num_escondida <- num_escondida
  arq$funcao_ativacao <- funcao_ativacao
  arq$der_funcao_ativacao <- der_funcao_ativacao

  # Pesos conectando entrada e a camada escondida
  num_pesos_entrada_escondida <- (num_entrada + 1) * num_escondida
  arq$pesos_escondida <- matrix(runif(min = -0.5, max = 0.5,
                                      num_pesos_entrada_escondida),
                                nrow = num_escondida, ncol = num_entrada + 1)

  # Pesos concectando a camada escondida e a camada de saida
  num_pesos_escondida_saida <- (num_escondida + 1) * num_entrada
  arq$pesos_saida <- matrix(runif(min = -0.5, max = 0.5,
                                  num_pesos_escondida_saida),
                            nrow = num_entrada, ncol = num_escondida + 1)

  return(arq)
}

# Reconstrucao
mlp.reconstruir <- function(arq, exemplo) {
  v_escondida <- arq$pesos_escondida %*% as.numeric(c(exemplo, 1))
  y_escondida <- arq$funcao_ativacao(v_escondida)

  v_saida <- arq$pesos_saida %*% c(y_escondida, 1)
  y_saida <- arq$funcao_ativacao(v_saida)

  resultado <- list()
  resultado$v.escondida <- v_escondida
  resultado$y_escondida <- y_escondida
  resultado$v_saida <- v_saida
  resultado$y_saida <- y_saida

  return(resultado)
}

# Back Propagation
mlp.retropropagacao <- function(arq, dados, n, limiar) {
  entropia_cruzada <- 2 * limiar
  entropia_cruzada_anterior <- 0
  epocas <- 0

  while (abs(entropia_cruzada - entropia_cruzada_anterior) > limiar) {
    entropia_cruzada_anterior <- entropia_cruzada
    entropia_cruzada <- 0

    # Epoch
    for (i in 1:nrow(dados)) {
        # Dados de treinamento
        x <- dados[i, ]

        # Saida da rede
        resultado <- mlp.reconstruir(arq, x)
        y <- resultado$y_saida

        # Erro
        erro <- x - y

        entropia <- (1 - x) * log(1 - y)
        if (any(is.nan(entropia))) {
           print(i)
        }
        entropia_cruzada <- entropia_cruzada + -sum((x * log(y) + (1 - x) * log(1 - y)))

        # Gradiente local dos neuronios de saida
        grad_local_saida <- erro * arq$der_funcao_ativacao(y)

        # Gradiente local dos neuronios escondidos
        pesos_saida <- arq$pesos_saida[, 1:arq$num_escondida] # Ignorar bias
        grad_local_escondida <- t(arq$der_funcao_ativacao(resultado$y_escondida)) *
                                (t(grad_local_saida) %*% pesos_saida)

        # Ajuste dos pesos
        arq$pesos_saida <- arq$pesos_saida + n *
                           (grad_local_saida %*% c(resultado$y_escondida, 1))
        # Neuronios escondidos
        arq$pesos_escondida <- arq$pesos_escondida + n *
                               (t(grad_local_escondida) %*% as.numeric(c(x, 1)))
    } # Fim da epoca

    entropia_cruzada <- entropia_cruzada / nrow(dados)
    print(entropia_cruzada)
    epocas <- epocas + 1
  }

  retorno <- list()
  retorno$arq <- arq
  retorno$epocas <- epocas
  return(retorno)
}

mlp.testa.digitos <- function(modelo,
                              caminho.mnist,
                              digitos = c(1, 3, 4, 7, 9),
                              taxa.ruido = 0.1) {
  digitos.teste <- carrega.digitos(caminho.mnist, digitos, 1, 0)

  plotdim <- 2 * orgdim
  plot(c(1, (plotdim[1] + 5) * length(digitos)),
       c(1, (plotdim[2] + 5) * 3),
       type = "n", xlab = "", ylab = "")

  x <- 1

  for (i in 1:nrow(digitos.teste)) {
    padrao <- digitos.teste[i, ]

    ruido <- (runif(length(padrao), 0, 1) > taxa.ruido) * 1
    entrada <- padrao * ruido
    resultado <- mlp.reconstruir(modelo$arq, entrada)
    y <- resultado$y_saida
    y <- as.vector(ifelse(y >= 0.5, 1, 0))

    # Padrao original
    img <- padrao;
    dim(img) <- orgdim
    image <- as.raster( (img + 1) / 2)
    rasterImage(image, x, 1, x + plotdim[1], plotdim[2], interpolate = F)

    # Entrada com ruido
    img <- entrada;
    dim(img) <- orgdim
    image <- as.raster( (img + 1) / 2)
    rasterImage(image, x, 1 + (plotdim[2] + 5),
                x + plotdim[1], 1 + 2 * (plotdim[2] + 5),
                interpolate = F)

    # Imagem recuperada
    img <- y;
    dim(img) <- orgdim
    image <- as.raster( (img + 1) / 2)
    rasterImage(image, x, 1 + 2 * (plotdim[2] + 5),
                x + plotdim[1], 1 + 2 * (plotdim[2] + 5) + plotdim[2],
                interpolate = F)

    x <- x + plotdim[1] + 5
  }
}
