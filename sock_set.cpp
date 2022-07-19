#include "header/sockset.h"


void Sock_set::Set_listen_fd()
{
    if((listen_fd = socket(AF_INET, SOCK_STREAM, 0))==-1)
        exit(1);
}


void Sock_set::Bind_addr()
{
    if(bind(listen_fd, (struct sockaddr *)&addr, addrlen) == -1)
        exit(1);
}

void Sock_set::Listen()
{
    listen(listen_fd, 5);
}