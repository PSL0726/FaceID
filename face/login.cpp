#include "login.h"
#include "ui_login.h"
#include "signup.h"
#include "chat.h"
#include <QMessageBox>
#include <iostream>
#include <unistd.h>
#include <cstring>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/types.h>

login::login(int sock,QWidget *parent) :
    QDialog(parent),
    ui(new Ui::login)
{
    this->sock = sock;
    ui->setupUi(this);
}

login::~login()
{
    delete ui;
}



void login::on_login_btn_clicked()
{
    QString id = ui->id_input->text();
    QString pw = ui->pw_input->text();

    if(id == "" || pw == "")
    {
        QMessageBox::information(this, "error", "공백");
    }
    else
    {
        char msg[1024];
        std::string send_data = "login/" + id.toStdString() + "/" + pw.toStdString();
        write(sock, send_data.c_str(), sizeof(send_data));
        read(sock, msg, sizeof(msg));
        qDebug()<<msg;
        if(strcmp(msg, "OK")==0)
        {
            this->hide();
            chat ch;
            ch.setModal(true);
            ch.exec();
            this->show();
        }
        else
            QMessageBox::information(this, "error", "ID/PW CHECK!");
    }
}

void login::on_out_btn_clicked()
{
    exit(0);
}

void login::on_signup_btn_clicked()
{
    this->hide();
    signup signup(sock);
    signup.setModal(true);
    signup.exec();
    this->show();
}
