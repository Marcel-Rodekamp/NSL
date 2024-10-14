set terminal postscript landscape enhanced color
set encoding iso_8859_1
set grid
set samples 400
set fit errorvariables
set fit results
set print "fit_masses.out"

set xlabel "m_s"
set ylabel "t"
set key left
set ytics format "%g"

##################################################
# Relevant code starts here.

# Fitting the solver time depending on the staggered mass:
set output "graphs/time.eps"
mytime(x)=t0*((t1+x)/(t2+x))
t0=0.01
t1=5
t2=0.1
set fit prescale
fit [:] mytime(x) "meas/time.csv" u 1:3:5 yerror via t0, t1, t2
plot "meas/time.csv" u 1:3:5 w yerrorbars title "data", mytime(x) title "fit"

# Shift eigenvalues by minimal staggered mass
if(!exists("mass_shift")) mass_shift=0
t1=t1+mass_shift
t2=t2+mass_shift

# Fitting the force between two mass terms:
set ylabel "m_s"
set zlabel "F"
set output "graphs/small_f.eps"
small(x,y)=a*abs(x-y)/((b+y)*(b+x))**c
a=0.5
b=t2
c=0.4
set fit prescale
fit small(x,y) "meas/small_f.csv" u 1:2:4:6 zerror via a,b,c
splot "meas/small_f.csv" u 1:2:4 title "data", small(x,y) title "fit"

# Fitting the remaining force depending on the staggered mass:
set ylabel "F"
set output "graphs/large_f.eps"
large(x)=A1-A2*x/(B*(B+x))**C
A1=10
A2=1
B=b
C=0.8
set fit prescale
fit [0.01:] large(x) "meas/large_f.csv" u 1:3:5 yerror via A1,A2,B,C
plot "meas/large_f.csv" u 1:3:5 w yerrorbars title "data", large(x) title "fit"

# Calculating masses and time scales used for the rest of the simulations:
diff=t1/t2
m1=t2*diff**(1./3)
m2=t2*diff**(2./3)
N1=sqrt(large(m2)/small(0,m1))
N2=sqrt(large(m2)/small(m1,m2))
N=floor(sqrt(N1*N2))
if(N==0) N=1
if(m2>m1 && m1>mass_shift){
	print sprintf("%d %d", N, N)
	print sprintf("%g %g", m1, m2)
}else{
	print sprintf("%d %d", 1, 1)
	print sprintf("%g %g", mass_shift+0.3, mass_shift+1)
}

# Relevant code ends here.
#################################################

# From here on some interesting but not important and not used information is printed.

print ""
print diff
print diff**(1./2)
print diff**(1./3)
print diff**(1./4)
print diff**(1./5)

print ""
print sprintf("1\n%g\n%g", sqrt(large(t2*sqrt(diff))/small(0,t2*sqrt(diff))), t2*sqrt(diff))

print ""
print sprintf("2\n%g\t%g\n%g\t%g", sqrt(large(t2*diff**(2./3))/small(0,t2*diff**(1./3))), sqrt(large(t2*diff**(2./3))/small(t2*diff**(1./3),t2*diff**(2./3))), t2*diff**(1./3), t2*diff**(2./3))

print ""
print sprintf("3\n%g\t%g\t%g\n%g\t%g\t%g", sqrt(large(t2*diff**(3./4))/small(0,t2*diff**(1./4))), sqrt(large(t2*diff**(3./4))/small(t2*diff**(1./4),t2*diff**(2./4))), sqrt(large(t2*diff**(3./4))/small(t2*diff**(1./4),t2*diff**(3./4))), t2*diff**(1./4), t2*diff**(2./4), t2*diff**(3./4))

print ""
print sprintf("4\n%g\t%g\t%g\t%g\n%g\t%g\t%g\t%g", sqrt(large(t2*diff**(4./5))/small(0,t2*diff**(1./5))), sqrt(large(t2*diff**(4./5))/small(t2*diff**(1./5),t2*diff**(2./5))), sqrt(large(t2*diff**(4./5))/small(t2*diff**(2./5),t2*diff**(3./5))), sqrt(large(t2*diff**(4./5))/small(t2*diff**(3./5),t2*diff**(4./5))), t2*diff**(1./5), t2*diff**(2./5), t2*diff**(3./5), t2*diff**(4./5))
