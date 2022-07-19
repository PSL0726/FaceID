#ifndef __SERVER_H_
#define __SERVER_H_

#include <iostream>
#include <stdlib.h>
#include <unistd.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/epoll.h>
#include <string.h>
#include "event.h"
#include "recv.h"
#include "udata.h"


class Server : public Event, public Recv
{
    public:
        Server(int port_num) : port(port_num) 
        {}
        ~Server() {}
        void Main_Func();
        void Accept();
        void Disconnect();
        
    private:
        struct sockaddr_in client_addr;
        socklen_t clnt_len;
        udata *user_data;
        int port, client_fd, read_cnt, i;
        char buf[SIZE_CONST::BUF_SIZE];      
};
#endif