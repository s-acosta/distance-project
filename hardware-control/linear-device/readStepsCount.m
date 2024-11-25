function [src, stop] = readStepsCount(src, ~)

% Read the ASCII data from the serialport object.
data = readline(src);

if data == "S"
   configureCallback(src, "off");
end

if ~any(regexp(data ,'[0-9]'))
    stop = 1;
%     fprintf(strcat(data, ' \n'))
    return
else
    stop = 0;
    count_marker_position = strfind(data{1}, 'c');
    time_marker_position = strfind(data{1}, 't');

    steps = str2double(data{1}(count_marker_position+1:time_marker_position-1));
    time = str2double(data{1}(time_marker_position+1:end));
    
    src.UserData.Time(end+1) = time;
    src.UserData.Steps(end+1) = steps;

    % fprintf('Time is %i miliseconds \n', time);
    % fprintf('Steps are %.2f \n', steps)

end

end