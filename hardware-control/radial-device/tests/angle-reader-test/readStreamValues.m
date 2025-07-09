function src = readStreamValues(src, ~)

% Read the ASCII data from the serialport object.
data = readline(src);

if data == "S"
   configureCallback(src, "off");
end

if ~any(regexp(data ,'[0-9]'))
    fprintf(strcat(data, ' \n'))
    return
end

marker = data{1}(1);
data = str2double(data{1}(2:end));

switch marker
    case 't'
        time = data;
%          fprintf('Time is %i miliseconds \n', time);
        src.UserData.Time(end+1) = time;
    case 'c'
        angle = 0.375 * data;
        fprintf('Steps are %.2f \n', data)
        src.UserData.Angle(end+1) = angle;
end

% Convert the string data to numeric type and save it in the UserData
% property of the serialport object.


end