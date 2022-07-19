#ifndef DATABASE_H
#define DATABASE_H
#include <QSqlDatabase>
#include <QSqlQuery>
#include <QDebug>
#include <QtSql>
#include <QSqlTableModel>

class Database
{
private:
    QSqlDatabase db;
public:
    Database() {
        db = QSqlDatabase::addDatabase("QMYSQL");
        db.setHostName("127.0.0.1");      // IP 또는 DNS Host name
        db.setDatabaseName("face"); // DB명
        db.setUserName("root");     // 계정 명
        db.setPassword("iot");     // 계정 Password

        if(!db.open())
        {
            qDebug()<<"connect error";
            exit(1);
        }
    }
    ~Database() {
        db.close();
    }
};

#endif // DATABASE_H
