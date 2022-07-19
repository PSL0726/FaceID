#ifndef __DATABASE_H_
#define __DATABASE_H_
#include "/usr/include/mysql/mysql.h"
#include <iostream>
#include <cstring>

class Database
{
protected:
    MYSQL *connection = NULL, conn;
    MYSQL_RES *sql_result;
    MYSQL_ROW sql_row;


public:
    Database()
    {
        mysql_init(&conn);
        connection = mysql_real_connect(&conn, "localhost", "root", "iot", "face", 3306, NULL, 0);
        if(connection==NULL)
        {
            fprintf(stderr, "Failed to connect to databases: Error: %s\n",
            mysql_error(&conn));
            return;
        }
        else
        {
            puts("Database connect!");
        }
    }
};
#endif