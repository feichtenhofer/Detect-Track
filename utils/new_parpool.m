function p = new_parpool(number)    
    
    if ~exist('number', 'var')
        number = cpu_cores();
    end

    if ~isempty(gcp('nocreate'))
      p = gcp;
%         delete(gcp);
    else
      p = parpool(number);   
    end
end