#ifndef LOGIN_H
#define LOGIN_H

#include <QDialog>

namespace Ui {
class login;
}

class login : public QDialog
{
    Q_OBJECT

public:
    explicit login(int sock, QWidget *parent = nullptr);
    ~login();

private slots:

    void on_login_btn_clicked();

    void on_out_btn_clicked();

    void on_signup_btn_clicked();

private:
    Ui::login *ui;
    int sock;
};

#endif // LOGIN_H
