% output

discard = 10000;
% if running on results directly - define kl,kr,J,I

outputl = zeros(kl,2);
outputr = zeros(kr,2);
outputsig = zeros(3,2);
outputdesc = zeros(7,2);

for i=1:kl,
    outputl(i,:) = [mean(betasave2(discard+1:J,i))    std(betasave2(discard+1:J,i))];
    if abs(outputl(i,1))<0.001,
        disp(outputl(i,:));
    end;
end;

for i=kl+1:kl+kr,
    outputr(i-kl,:) = [mean(betasave2(discard+1:J,i))    std(betasave2(discard+1:J,i))];
    if abs(outputr(i-kl,1))<0.001,
        disp(outputr(i-kl,:));
    end;
end;

for i=1:3,
    outputsig(i,:) = [mean(SIGsave(discard+1:J,i))    std(SIGsave(discard+1:J,i))];
    if abs(outputsig(i,1))<0.001,
        disp(outputsig(i,:));
    end;
end;

for i=1:7,
    outputdesc(i,:) = [mean(descsave(discard+1:J,i))    std(descsave(discard+1:J,i))];
    if abs(outputdesc(i,1))<0.001,
        disp(outputdesc(i,:));
    end;
end;

disp('obs');
disp(I);
disp('Betal')
disp(outputl)
disp('Betar')
disp(outputr)
disp('SIG')
disp(outputsig)
disp('desc stats')
disp(outputdesc)
