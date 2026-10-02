function [x,bits] = qpsk_symbols(labels)
% Gray cycle: 0=00, 1=01, 2=11, 3=10, unit symbol energy.
constellation = [1+1i;-1+1i;-1-1i;1-1i]/sqrt(2);
mapping = uint8([0 0;0 1;1 1;1 0]);
x = reshape(constellation(double(labels)+1),size(labels));
bits = reshape(mapping(double(labels(:))+1,:).',2,[]);
end
