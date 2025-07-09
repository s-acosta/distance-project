encoder_port = 'com5';
encoder = serialport(encoder_port, 115200);
configureTerminator(encoder, "LF");
encoder.UserData = struct("Angle",[],"Time", [], "Count",1);
while ~contains(encoder.readline(),  "Arduino is ready")
end
tic
configureCallback(encoder,"terminator",@readStreamValues);

%%

motor = serialport("COM3", 115200);
configureTerminator(motor, "LF");
while ~contains(motor.readline(),  "Arduino is ready")
end

%%
order = strcat("<", "1", ">");
motor.writeline(order);
motor.writeline("0, 0");
while ~contains(motor.readline(),  "Movement Done")
end

pause(1);

for i = 1:10
    while toc < 5*(i-1)
    end
    motor.writeline("1, 250");
    while ~contains(motor.readline(),  "Movement Done")
    end
    pause(1);
    motor.writeline("0,0")
    while ~contains(motor.readline(),  "Movement Done")
    end
    pause(1);

end

configureCallback(encoder, "off");

%%
figure,
plot(encoder.UserData.Time, encoder.UserData.Angle)
xlabel('Time (ms)')
ylabel('Angle')
title('Moving Arm')