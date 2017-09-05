# Como rodar

## Docker

Primeiro é só instalar o Docker como preferir. Dependendo da distribuição, talvez precise rodar os comandos como sudo e inicializar serviços (mas acho que o Ubuntu faz isso automático, mas precisei manualmente fazer isso no Arch)

Depois é só rodar `docker pull adnrv/opencv`, talvez precise de sudo, e ele vai baixar a imagem do Adin.

### Debugando docker

Se quiser rodar separado um shell para o docker, pra entender melhor as coisas, abrir o shell python ou algo assim, só rodar `sudo docker run -it adnrv/opencv bash`.

## Programa

Pra rodar o programa, depois disso, é só dar make dentro da pasta. Se quiser ver o comando que roda é só ver o Makefile. Novamente, talvez precise de sudo.

Se der um erro `Gtk-WARNING **: cannot open display:` só usar `xhost local:root`
