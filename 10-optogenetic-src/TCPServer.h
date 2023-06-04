#pragma once
#include <WINSOCK2.H>


#define _WINSOCK_DEPRECATED_NO_WARNINGS

class TCPServer
{
public:

	//int result = 0;

	SOCKET server_ = INVALID_SOCKET; //socket 
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
