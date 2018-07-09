source("autoencoder.R")
source("utils.R")

digitos_10_semruido <- carrega.digitos("mnist_png",
                                       c(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
                                       10, 1, 0)
digitos_50_semruido <- carrega.digitos("mnist_png",
                                       c(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
                                       50, 1, 0)
digitos_100_semruido <- carrega.digitos("mnist_png",
                                       c(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
                                       100, 1, 0)

dimensao_img <- dim(digitos_10_semruido)[2]

# Autoencoder
arq_autoencoder1 <- autoencoder.arquitetura(dimensao_img,
                                            dimensao_img + 30,
                                            funcao_ativacao,
                                            der_funcao_ativacao)
arq_autoencoder2 <- autoencoder.arquitetura(dimensao_img,
                                            dimensao_img + 60,
                                            funcao_ativacao,
                                            der_funcao_ativacao)
arq_autoencoder3 <- autoencoder.arquitetura(dimensao_img,
                                            dimensao_img + 90,
                                            funcao_ativacao,
                                            der_funcao_ativacao)
