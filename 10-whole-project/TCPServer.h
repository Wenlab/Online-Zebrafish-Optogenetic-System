#pragma once
//https://blog.csdn.net/starelegant/article/details/72460728  
//避免版本不同造成的重定义错误
#include <WINSOCK2.H>


#define _WINSOCK_DEPRECATED_NO_WARNINGS

class TCPServer
{
public:

	//int result = 0;

	SOCKET server_ = INVALID_SOCKET; //socket 对象
	WSADATA data_;
	SOCKADDR_IN addrClient;
	SOCKET socketConn2;
	int data = 0;

	void initialize();

	int waitingConnect();

	int receive();

	void sendData();

	void close();


private:

};

int string2int(char *arg);
