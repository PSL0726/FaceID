#ifndef SIGNUP_H
#define SIGNUP_H

#include <database.h>
#include <cstring>
#include <QDialog>
#include <QtWidgets>

namespace Ui {
class signup;
}

class signup : public QDialog
{
    Q_OBJECT

public:
    explicit signup(int sock, QWidget *parent = nullptr);
    ~signup();

private slots:
    void on_out_btn_clicked();

    void on_id_check_clicked();

    void on_singup_btn_clicked();

private:
    Ui::signup *ui;
    int sock;
    bool check;
};

#endif // SIGNUP_H
