% --- Configure serial once ---
port = "COM3";        % <-- change to your Teensy port
baud = 115200;
s = serialport(port, baud);
s.OutputBufferSize = 300000;             % big enough for a full frame
configureTerminator(s, "LF");
flush(s);

%% --- Load MATLAB demo image "peppers.png" ---
raw = imread("onion.png");

% --- Resize to 240x240 and ensure RGB uint8 ---
H = 240; W = 240;
img = imresize(raw, [H W]);
img = uint8(img(:,:,1:3));   % make sure it's RGB

% --- Convert to RGB565 (5-6-5 format) ---
R5 = bitshift(uint16(img(:,:,1)), -3);
G6 = bitshift(uint16(img(:,:,2)), -2);
B5 = bitshift(uint16(img(:,:,3)), -3);
rgb565 = bitor( bitshift(R5,11), bitor(bitshift(G6,5), B5) );

% --- Make big-endian byte stream [HI LO HI LO ...] ---
hi = uint8(bitshift(rgb565, -8));
lo = uint8(bitand(rgb565, 255));
bytes = reshape([hi(:)'; lo(:)'], 1, []);

% --- Send header and data to Teensy ---
header = sprintf("IMG %d %d\n", W, H);
write(s, header, "string");
write(s, bytes, "uint8");

% --- Teensy reply (optional) ---
pause(0.05);
if s.NumBytesAvailable > 0
    disp(readline(s));
end


%%
raw = imread("peppers.png");

% --- Resize to 240x240 and ensure RGB uint8 ---
H = 240; W = 240;
img = imresize(raw, [H W]);
if ~isa(img,'uint8'); img = im2uint8(img); end
if ndims(img)==2; img = repmat(img,[1 1 3]); end
img = img(:,:,1:3);

% --- RGB565 (HxW uint16) ---
R5 = bitshift(uint16(img(:,:,1)), -3);
G6 = bitshift(uint16(img(:,:,2)), -2);
B5 = bitshift(uint16(img(:,:,3)), -3);
rgb565 = bitor( bitshift(R5,11), bitor(bitshift(G6,5), B5) );

% --- Row-major LITTLE-ENDIAN payload ---
frameLE = rgb565.';                      % W x H -> row-major when linearized
payload  = typecast(frameLE(:), 'uint8');% LITTLE-endian on x86 by default (good)

% --- Tell Teensy what's coming and send it ---
writeline(s, sprintf('IMG %d %d', W, H));
write(s, payload, "uint8");              % single big write = fastest
%%
% --- Open serial once (adjust port) ---
s.OutputBufferSize = 300000;         % big enough for a whole frame

raw = imread("onion.png");

% --- Resize to 240x240 and ensure RGB uint8 ---
H = 240; W = 240;
img = imresize(raw, [H W]);
if ~isa(img,'uint8'); img = im2uint8(img); end
if ismatrix(img); img = repmat(img,[1 1 3]); end
img = img(:,:,1:3);                  % ensure RGB

% --- Convert to RGB565 (HxW uint16) ---
R5 = bitshift(uint16(img(:,:,1)), -3);
G6 = bitshift(uint16(img(:,:,2)), -2);
B5 = bitshift(uint16(img(:,:,3)), -3);
rgb565 = bitor( bitshift(R5,11), bitor(bitshift(G6,5), B5) );

% --- Make row-major, big-endian payload efficiently ---
frame = rgb565.';                    % W x H; now linearization is row-major
payload = typecast(swapbytes(frame(:)), 'uint8');  % big-endian [HI LO HI LO ...]

% --- Send header and payload ---
writeline(s, sprintf('IMG %d %d', W, H));   % your sketch expects this line
write(s, payload, "uint8");                 % one big write = fastest

% --- (Optional) read Teensy reply ---
if s.NumBytesAvailable > 0
    disp(readline(s));               % prints "OK" if your sketch sends it
end

%%
raw = imread("onion.png");

% --- Resize to 240x240 and ensure RGB uint8 ---
H = 240; W = 240;
img = imresize(raw, [H W]);
if ~isa(img,'uint8'); img = im2uint8(img); end
if ismatrix(img); img = repmat(img,[1 1 3]); end
img = img(:,:,1:3);

% --- RGB565 (HxW uint16) ---
R5 = bitshift(uint16(img(:,:,1)), -3);
G6 = bitshift(uint16(img(:,:,2)), -2);
B5 = bitshift(uint16(img(:,:,3)), -3);
rgb565 = bitor( bitshift(R5,11), bitor(bitshift(G6,5), B5) );

% --- Row-major LITTLE-ENDIAN payload ---
frameLE = rgb565.';                      % W x H -> row-major when linearized
payload  = typecast(frameLE(:), 'uint8');% LITTLE-endian on x86 by default (good)

% --- Tell Teensy what's coming and send it ---
writeline(s, sprintf('IMG %d %d', W, H));
write(s, payload, "uint8");              % single big write = fastest

% Optional: ack
% if s.NumBytesAvailable > 0, disp(readline(s)); end

%%
function upload_image(filename, screen_size, serial_obj)
    % Read the image
    img = imread(filename);
    
    % Resize image while keeping aspect ratio (similar to PIL thumbnail)
    img = imresize(img, screen_size, 'Antialiasing', true);
    
    % Get new size
    [img_h, img_w, ~] = size(img);
    
    % Calculate margins for centering
    top_margin = floor((screen_size(2) - img_h) / 2);
    bottom_margin = screen_size(2) - img_h - top_margin;
    left_margin = floor((screen_size(1) - img_w) / 2);
    right_margin = screen_size(1) - img_w - left_margin;
    
    % Send command to indicate image transfer
    header = sprintf("IMG %d %d\n", 240, 240);
    write(serial_obj, header, "string");
    
    % Send top margin rows
    for row = 1:top_margin
        write(serial_obj, zeros(1, screen_size(1)*2, 'uint8'), "uint8");
    end
    
    % Send image rows
    for row = 1:img_h
        % Left margin
        if left_margin ~= 0
            write(serial_obj, zeros(1, left_margin*2, 'uint8'), "uint8");
        end
        
        % Convert each pixel to 16-bit RGB565
        for col = 1:img_w
            r = bitshift(img(row,col,1), -3); % 5 bits
            g = bitshift(img(row,col,2), -2); % 6 bits
            b = bitshift(img(row,col,3), -3); % 5 bits
            
            % Pack into two bytes (RGB565)
            byte1 = bitor(bitshift(bitand(g, 0x07), 5), r);
            byte2 = bitor(bitshift(b, 3), bitshift(g, -3));
            
            write(serial_obj, [byte1 byte2], "uint8");
        end
        
        % Right margin
        if right_margin ~= 0
        write(serial_obj, zeros(1, right_margin*2, 'uint8'), "uint8");
        end

    end
    
    % Send bottom margin rows
    for row = 1:bottom_margin
        write(serial_obj, zeros(1, screen_size(1)*2, 'uint8'), "uint8");
    end
end

% Example usage
serial_obj = s;
upload_image("peppers.png", [240 240], serial_obj);
