#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include "login.h"
#include <QApplication>
#include <QMessageBox>
#include <QDebug>

int main(int argc, char *argv[])
{

    int sock = 0;
    struct sockaddr_in serv_addr;
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    sock = socket(AF_INET, SOCK_STREAM, 0);
    memset(&serv_addr, 0, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = inet_addr("127.0.0.1");
    serv_addr.sin_port = htons(atoi("9011"));

    if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) == -1)
    {
        QMessageBox msgBox;
        msgBox.setText("Connect Error");
        msgBox.exec();
        exit(1);
    }
    qDebug()<<sock;
    //write(sock,"serv",sizeof("serv"));
    QApplication a(argc, argv);
    login lo(sock);
    lo.show();

    return a.exec();
}
