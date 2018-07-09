library(png)

# Carrega os digitos da base MNIST
carrega.digitos <- function(caminho,
                            digitos = c(1, 3, 4, 7, 9), quantidade = 100,
                            treino = 1, taxa.ruido = 0.1) {
    padroes <- list()
    orgdim <<- NULL

    cat("\nCarregando digitos\n")
    for (i in digitos) {
       if (treino == 1) {
            caminho_arquivos <- paste(caminho, "/training/",
                                      as.character(i), sep = "")
        } else {
            caminho_arquivos <- paste(caminho, "/training/",
                                      as.character(i), sep = "")
        }

        arquivos <- list.files(caminho_arquivos)[1:quantidade]
        for (j in 1:quantidade) {
            caminho_img <- paste(caminho_arquivos, "/",
                                 arquivos[j], sep = "")
            img <- readPNG(caminho_img)
            orgdim <<- dim(img)
            dim(img) <- NULL
            padroes[[length(padroes) + 1]] <- as.vector(ifelse(img >= 0.2, 1, 0))
        }
    }

    padroes_matrix <- matrix(unlist(padroes),
                             ncol = orgdim[1] * orgdim[2],
                             byrow = TRUE)

    return(padroes_matrix)
}