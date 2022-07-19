#ifndef THREAD_H
#define THREAD_H
#include <QThread>
#include <QtDebug>
#include <cstring>

class Thread: public QThread
{
    Q_OBJECT

public:
    Thread(int sock, QObject *parent=0);

private:
    void run();
    int sock;
signals:
    void set_value(QString msg);
    void set_msg(QString msg);
};

#endif // THREAD_H
