#ifndef __ADDR_H_
#define __ADDR_H_

#include <sys/socket.h>
#include <arpa/inet.h>
#include <stdlib.h>

class Addr
{
    public:
        Addr() {}
        ~Addr() {}
        void Set_addr(int port);

    protected:
        struct sockaddr_in addr;
        socklen_t addrlen;

};

#endif