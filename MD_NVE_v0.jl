# This codes solve MD for a gas using a NVE ensemble
# This code is based on the preprint:
# https://arxiv.org/pdf/2103.16944.pdf
#-------------------------------------------------
using Base.Threads
using Random
Random.seed!(1234);
#-------------------------------------------------
@views function MD_main()
    ndims  = 3         # spatial dimensions
    nt     = 200       # steps
    N      = 100       # n atoms
    dt     = 1.E-4     # dt 
    Tfix   = 300.0     # target temperature
    sig    = 1.0       # L-J param 
    eps    = 1.0       # L-J param 
    r_ctf  = 2.5*sig   # cut-off radius
    u_ctf  = 4.0*eps*((sig/r_ctf)^12 - (sig/r_ctf)^6)          # L-J param (potential at cut-off)
    du_ctf = 24.0*eps*(-2.0*sig^12/r_ctf^13 + sig^6/r_ctf^7)   # L-J param (bend potential at cut-off radius)
    Lx     = 10.0*sig  # box side
    vol    = Lx^3      # volume
    rho    = N/vol     # density
    ign    = 20        # ignored steps
    #-------------------------------------------------
    # Initiate positions , velocities , & accelerations
    pos = zeros(N, ndims)
    vel = zeros(N, ndims) 
    acc = zeros(N, ndims)
    for i=1:N
        for k=1:ndims
            pos[i,k] = Lx*rand()
            vel[i,k] = rand()
        end
    end
    #--------------------------------------
    # Scale positions to the box size: coordinates will range between 0 and 1
    pos    .= pos./Lx 
    r_ctf2  = r_ctf*r_ctf # square of cutoff distance
    #----------------------------
    # Translate positions to the center-of-mass (com) reference frame.
    com = zeros(ndims)
    for k=1:ndims
        for i=1:N
            com[k] += pos[i,k]
            com[k] = com[k]/N
        end
    end
    for k=1:ndims
        pos[:,k] .-= com[k]
    end
    #---------------------
    P_S   = 0.0 # sum of pressures
    k_S   = 0.0 # sum of kinetic energies
    p_S   = 0.0 # sum of potential energies
    T_S   = 0.0 # sum of temperatures
    R     = zeros( N, N, ndims )  # inter-particle distances scaled to box size
    r     = zeros( N, N, ndims )  # inter-particle distances 4 real
    Tvec  = zeros(nt)
    Pvec  = zeros(nt)
    Zvec  = zeros(nt)
    ekvec = zeros(nt)
    epvec = zeros(nt)
    #------------------------------------
    # The time evolution loop ( main loop)
    for it=1:nt
        global P_S = P_S+1
        #-----------------------------
        # Periodic boundary conditions
        pos[pos .>  0.5] .-= 1
        pos[pos .< -0.5] .+= 1
        # Loop for computing forces
        pot = zeros(N)        # potential
        vrl = 0.0             # virial
        acc .= 0              # reset accelerations
        for i=1:N
            for j=i+1:N
                if j != i
                    r2 = zeros(N, N)
                    for k=1:ndims
                        R[i,j,k] = pos[i,k] - pos[j,k]
                        if abs(R[i,j,k]) > 0.5
                            R[i,j,k] -= sign(R[i,j,k]) # treatment of periodic BCs
                        end
                        r[i,j,k] = Lx*R[i,j,k]       # scale by box size
                        r2[i,j] += r[i,j,k]*r[i,j,k] # square of distance
                    end
                    if r2[i,j] < r_ctf2
                        # Compute interaction using Lennard-Jones potential
                        r1      = sqrt(r2[i,j])
                        ri2     = 1.0/r2[i, j]
                        ri6     = ri2*ri2*ri2
                        ri12    = ri6*ri6
                        sig6    = sig^6
                        sig12   = sig^12
                        u       =  4.0*eps*( sig12*ri12 - sig6*ri6) - u_ctf - r1*du_ctf
                        du      = 24.0*eps*ri2*(2.0*sig12*ri12 - sig6*ri6) + du_ctf*sqrt(ri2)
                        pot[j] += u
                        vrl    -= du*r2[i, j]    # virial
                        for k=1:ndims
                            acc[i,k] += du*R[i,j,k] # particle 1
                            acc[j,k] -= du*R[i,j,k] # particle 2
                        end
                    end
                else
                    println("Should we ever be here?")
                end
            end
        end
        vrl = -vrl/ndims
        #------------------
        # Update positions
        pos .+= dt.*vel .+ 0.5* acc.*dt.*dt
        #----------------------------
        # Compute temperature
        kin = zeros(N)  # kinetic energy
        v2  = zeros(N)
        for j=1:N
            for k=1:ndims
                v2[j] += vel[j,k]* vel[j,k]*Lx*Lx
            end
            kin[j] = 0.5*v2[j]
        end
        k_AVG = sum(kin)/N        # average kinetic energy
        T_i   = 2.0*k_AVG/ndims   # instantaneous temperature
        B     = sqrt(T_0/T_i)     # rescaling factor
        #--------------------------------
        # Apply thermostat: Rescale & update the velocities according to velocity Verlet algorithm
        vel  .= B.*vel + 0.5.*dt.*acc
        vel .+= 0.5.*dt.*acc
        #---------------------
        # Re: Compute temperature after re-scaling
        kin = zeros(N)  # kinetic energy
        v2  = zeros(N)
        for j=1:N
            for k=1:ndims
                v2[j] += vel[j,k]* vel[j,k]*Lx*Lx
            end
            kin[j] = 0.5*v2[j]
        end
        k_AVG = sum(kin)/N        # average kinetic energy
        T_i   = 2.0*k_AVG/ndims   # instantaneous temperature
        B     = sqrt(T_0/T_i)     # rescaling factor
        #------------------
        p_AVG = sum(pot)/N        # average pressure
        e_AVG = k_AVG + p_AVG     # average total energy
        P     = rho*T_i + vrl/vol # pressure
        Z     = P*vol/(N*T_i)     # compressibility factor
        if it>ign
            global P_S += P
            global k_S += k_AVG
            global p_S += p_AVG
            global T_S += T_i
        end
        Tvec[it]  = T_i
        Pvec[it]  = P
        Zvec[it]  = Z
        ekvec[it] = k_AVG
        epvec[it] = p_AVG
        # Output
        if mod(it,10)==0
            println("Step ", it)
            println("Temperature ", T_i)
        end
    end
    # Visualize
    t = (1:nt)*dt
    p1 = plot(t[3:end], Tvec[3:end], legend=:false, linewidth=:3.0, framestyle=:box, xlabel="time", ylabel="Temperature")
    p2 = plot(t[3:end], Pvec[3:end], legend=:false, linewidth=:3.0, framestyle=:box, xlabel="time", ylabel="Pressure")
    display(plot(p1,p2))
end

@time MD_main()