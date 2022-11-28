%将五个数字(day hour minute second microsecond)的timestamp换算成一个数字

function t = caculateTimeStamp(time)
    t=time(:,2)*3600*1000+time(:,3)*60*1000+time(:,4)*1000+time(:,5);

end