#include "signup.h"
#include "ui_signup.h"
#include <QMessageBox>
#include <QPixmap>
#include <cstring>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <iostream>

signup::signup(int sock,QWidget *parent) :
    QDialog(parent),
    ui(new Ui::signup)
{
    this->sock=sock;
    ui->setupUi(this);
}

signup::~signup()
{
    delete ui;
}

void signup::on_out_btn_clicked()
{
    this->close();
}

void signup::on_id_check_clicked()
{
    QString id = ui->id_input->text();
    if(id == "")
        QMessageBox::information(this, "error", "공백");
    else
    {
        char msg[1024];
        std::string send_data = "idcheck/" + id.toStdString();
        write(sock, send_data.c_str(),sizeof(send_data));
        std::cout<<send_data<<std::endl;
        read(sock, msg, sizeof(msg));
        qDebug()<<msg;
        if(strcmp(msg, "OK") == 0)
        {
            check = true;
            QMessageBox::information(this, "OK", "통과");
        }
        else
            QMessageBox::information(this, "error", "중복");

    }
}

void signup::on_singup_btn_clicked()
{
    if(!check)
        QMessageBox::information(this, "error", "중복확인");
    else
    {
        QString pw = ui->pw_input->text();
        QString name = ui->name_input->text();
        QString phone = ui->phone_input->text();
        if(pw == "" || name == "")
            QMessageBox::information(this, "error", "공백");
        else
        {
            if(ui->pw_input->text() != ui->pw_check->text())
                QMessageBox::information(this, "error", "비밀번호 불일치");
            else
            {
                std::string send_data = "signup/" + ui->id_input->text().toStdString() + "/" +
                    pw.toStdString() + "/" + name.toStdString() + "/" + phone.toStdString();
                write(sock, send_data.c_str(), sizeof(send_data));
                QMessageBox::information(this, "축", "축) 가입");
                this->close();
            }
        }
    }
}
