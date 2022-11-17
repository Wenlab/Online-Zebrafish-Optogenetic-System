#include"TCPServer.h"

#include <string>                                                                        
#include <iostream>
#include <cstring>
#include <Ws2tcpip.h>

#pragma comment(lib,"ws2_32.lib")

using namespace std;

void TCPServer::initialize()
{
	int result = 0;
	result = WSAStartup(MAKEWORD(2, 2), &data_);//inital
	if (result != 0)
	{
		std::cout << "WSAStartup() init error " << GetLastError() << std::endl;
		//system("pause");
		return;
	}

	server_ = socket(AF_INET, SOCK_STREAM, 0);

	SOCKADDR_IN addrSrv;
	addrSrv.sin_family = AF_INET;
	addrSrv.sin_port = htons(11000);
	addrSrv.sin_addr.S_un.S_addr = htonl(INADDR_ANY);//ip port 

	result = bind(server_, (LPSOCKADDR)&addrSrv, sizeof(SOCKADDR_IN));

	if (result != 0)
	{	
		std::cout << "bind() error" << result;
		//system("pause");
		return;
	}
	result = listen(server_, 10);
	if (result != 0)
	{
		std::cout << "listen error" << result;
		//system("pause");
		return;
	}

	return;
}


int TCPServer::waitingConnect()
{
	SOCKADDR_IN addrClient;
	int len = sizeof(SOCKADDR);

	std::cout << "wait new connect......" << std::endl;
	socketConn2 = accept(server_, (SOCKADDR*)&addrClient, &len);
	if (socketConn2 == INVALID_SOCKET)
	{
		return 0;
	}

	int Timeout = 4;   //ms
	setsockopt(socketConn2, SOL_SOCKET, SO_RCVTIMEO, (char*)&Timeout, sizeof(int));

	if (socketConn2 == SOCKET_ERROR)
	{
		std::cout << " accept error" << WSAGetLastError();
		return 0;
	}
	std::cout << "connect successful" << std::endl;
	return 1;
}




int TCPServer::receive()
{
	char recvBuff[1024];
	memset(recvBuff, 0, sizeof(recvBuff));
	recv(socketConn2, recvBuff, sizeof(recvBuff), 0);

	//std::cout << "client say:" << recvBuff << std::endl;

	data = string2int(recvBuff);   
	//cout << data << endl;
	int is_ok;
	is_ok = (WSAECONNRESET != WSAGetLastError());

	Sleep(1);

	return is_ok;
}


void TCPServer::sendData()
{
	char sendData[100];
	std::string str = "aaaa";
	memset(sendData, 0, 100);
	memcpy(sendData, str.c_str(), sizeof(str));
	//cout << "send message to client" << endl;
	send(socketConn2, sendData, strlen(sendData), 0);

}


void TCPServer::close()
{
	closesocket(server_);
	WSACleanup();
}

int string2int(char *arg)
{
	char c;
	int i, j, tmp = 0, tmpc = 0;
	for (i = 0; i < strlen(arg); i++) {
		c = (*(arg + i));
		tmp = tmp * 10 + (int)c - 48;
	}
	return tmp;
}