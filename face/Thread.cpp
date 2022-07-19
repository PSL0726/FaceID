#include "Thread.h"
#include <unistd.h>
#include <QMessageBox>
#include <QDebug>
#include <cstring>
#include <string>
#include <string.h>

Thread::Thread(int sock, QObject *parent) :
    QThread(parent)
{
    this->sock=sock;
}
void Thread::run()
{
    while(1)
    {
        char msg[1024];
        memset(msg,0,sizeof(msg));
        read(sock, msg, sizeof(msg));
        qDebug()<<msg;

        if(strstr(msg,"상담요청"))
        {
            QMessageBox::information(nullptr,"알림","새로운 채팅요청이 도착했어요!");
            emit set_value(QString(msg)); // 창을 여는거라서 while로 돌리면 안됨
        }
        else
        {
              emit set_msg(QString(msg)); // tablewidget에 띄우는거라서 while해야함
        }                                 // 근데 내가 보낸내용도 써져서 지금 내 내용은 2번 적힘

    }
}
