function root_path = get_root_path()

if ispc
  root_path = 'datasets\ILSVRC2015\';
else
   root_path = 'data/ILSVRC'; 
end