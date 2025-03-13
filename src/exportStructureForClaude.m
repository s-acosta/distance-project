function exportStructureForClaude(data, filename)
% EXPORTSTRUCTUREFORCLAUDE Exports a MATLAB structure to a JSON file
% that can be easily uploaded to Claude for analysis.
%
% Usage:
%   exportStructureForClaude(myStructure, 'myStructureData.json')
%
% Inputs:
%   data - MATLAB structure/array/cell to be exported
%   filename - Output filename (should end with .json)
%
% Example:
%   myStruct.field1 = 'hello';
%   myStruct.field2 = [1 2 3 4];
%   myStruct.field3.subfield = {1, 2, 'text'};
%   exportStructureForClaude(myStruct, 'myData.json');
%   % Then upload myData.json to Claude

% Input validation
if nargin < 2
    error('Both data and filename arguments are required');
end

if ~ischar(filename) && ~isstring(filename)
    error('Filename must be a character array or string');
end

% Check if filename has .json extension
[~, ~, ext] = fileparts(filename);
if ~strcmpi(ext, '.json')
    filename = [filename '.json'];
    fprintf('Added .json extension to filename: %s\n', filename);
end

try
    % Convert to JSON with pretty formatting (easier to read)
    jsonStr = jsonencode(data, 'PrettyPrint', true);
    
    % Write to file
    fid = fopen(filename, 'w');
    if fid == -1
        error('Could not open file for writing: %s', filename);
    end
    fprintf(fid, '%s', jsonStr);
    fclose(fid);
    
    % Success message
    fprintf('Successfully exported structure to %s\n', filename);
    fprintf('You can now upload this file to Claude for analysis\n');
    
catch ME
    % Handle special cases of data types that might cause issues
    if contains(ME.message, 'encodej')
        fprintf('Error: Some data types in your structure may not be JSON-compatible.\n');
        fprintf('Common problematic types: function handles, objects, or sparse matrices.\n');
        fprintf('Attempting alternate export method for complex structures...\n');
        
        % Alternative approach for complex structures
        alternateExport(data, filename);
    else
        % Rethrow other errors
        rethrow(ME);
    end
end
end

function alternateExport(data, filename)
% Handle complex structures by converting problematic fields to strings
data = makeJSONCompatible(data);

% Try encoding again
try
    jsonStr = jsonencode(data, 'PrettyPrint', true);
    
    % Write to file
    fid = fopen(filename, 'w');
    fprintf(fid, '%s', jsonStr);
    fclose(fid);
    
    fprintf('Successfully exported modified structure to %s\n', filename);
    fprintf('Note: Some complex data types were converted to string representations\n');
catch ME
    fprintf('Error during alternate export: %s\n', ME.message);
    fprintf('Please try exporting specific fields of your structure separately\n');
end
end

function data = makeJSONCompatible(data)
% Recursively make a structure JSON compatible by converting
% problematic types to string representations

if isstruct(data)
    fields = fieldnames(data);
    for i = 1:numel(fields)
        for j = 1:numel(data)
            if isfield(data(j), fields{i})
                data(j).(fields{i}) = makeJSONCompatible(data(j).(fields{i}));
            end
        end
    end
elseif iscell(data)
    for i = 1:numel(data)
        data{i} = makeJSONCompatible(data{i});
    end
elseif isa(data, 'function_handle')
    data = char(data);
elseif isobject(data) && ~(ismatrix(data) && ~issparse(data))
    try
        % Try to convert to struct if possible
        data = struct(data);
        data = makeJSONCompatible(data);
    catch
        % Otherwise use string representation
        data = class(data);
    end
elseif issparse(data)
    % Convert sparse to full with a note
    data = struct('type', 'sparse_matrix', 'size', size(data), 'values', full(data));
elseif ~(isnumeric(data) || islogical(data) || ischar(data) || isstring(data))
    % Convert any other problematic types to their class name
    data = class(data);
end
end