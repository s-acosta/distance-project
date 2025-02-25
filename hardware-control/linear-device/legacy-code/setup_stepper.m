function stepper = setup_stepper(com_port)

    stepper = serialport(com_port, 115200);
    configureTerminator(stepper, "LF");
    
    while ~contains(stepper.readline(),  "Arduino is ready")
    end
    
    stepper.flush();
    stepper.UserData = struct("Steps",[],"Time", []);
    configureCallback(stepper,"terminator", @readStepsCount)

end