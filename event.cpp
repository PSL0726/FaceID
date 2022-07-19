#include "header/event.h"

void Event::Create_event()
{
    events = (struct epoll_event *)malloc(sizeof(struct epoll_event) * SIZE_CONST::EPOLL_SIZE);
    if((epoll_fd = epoll_create(100)) == -1)
        exit(1);
}

void Event::Set_event()
{
    ev.events = EPOLLIN;
    ev.data.fd = listen_fd;
    epoll_ctl(epoll_fd, EPOLL_CTL_ADD, listen_fd, &ev);
    memset(user_fds, -1, sizeof(int) * SIZE_CONST::BUF_SIZE);
}

void Event::E_wait()
{
    event_cnt = epoll_wait(epoll_fd, events, SIZE_CONST::EPOLL_SIZE, -1);
    if(event_cnt == -1)
        exit(1);
}