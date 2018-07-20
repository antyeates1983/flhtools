! FASTFLH.F90 - compute field-line helicity for regular cartesian data
!             - traces field lines with midpoint method
!             - uses openmp
!
!             - command line input:
!               [1] bfile (string) - unformatted file with magnetic field at face centres
!               [2] afile (string) - unformatted file with vector potential on edges
!               [3] hmax - (double) - max step size (as fraction of min cell size)
!
! anthony.yeates@durham.ac.uk
!
!================================================================

!****************************************************************
program fastflh
!----------------------------------------------------------------
! Read in magnetic field datacube and field-line startpoints, 
! trace field lines, and output complete field lines.
!----------------------------------------------------------------

    implicit none
    
    character(128) :: bfile, afile, hmax, maxerrors
    character*(*), parameter :: x0file='x0.unf', flhfile='flh.unf'
    integer :: nx, ny, nz, nl, i
    double precision, dimension(:), allocatable :: x, y, z, xc, yc, zc, flh
    double precision, dimension(:,:), allocatable :: x0
    double precision, dimension(:,:,:), allocatable :: bx, by, bz, ax, ay, az
    double precision, dimension(6) :: xmin, ymin, zmin, xmax, ymax, zmax
    double precision, dimension(3) :: k1, k2, dx1, dx2, x1, x2, xl
    double precision :: dx, dy, dz, ddirn, maxds, ds_dt, error, flh0, flh1
    double precision :: maxerror, ds, ds1, ds2, ds3
    integer :: dirn, nxt
    integer, parameter :: nmax=100000 
    double precision, parameter :: minB=1d-4

    !-------------------------------------------------------------
    ! Get command line arguments:
    call get_command_argument(1, bfile)
    call get_command_argument(2, afile)
    call get_command_argument(3, hmax)
    call get_command_argument(4, maxerrors)
    read(maxerrors,*) maxerror
  
    !-------------------------------------------------------------
    ! Read B from file:
    open(1, file=trim(bfile), form='unformatted')
    ! - grid dimensions:
    read(1) nx
    read(1) ny
    read(1) nz
    ! - coordinate arrays at grid points:
    allocate(x(0:nx-1), y(0:ny-1), z(0:nz-1))
    read(1) x
    read(1) y
    read(1) z
    ! - coordinate arrays at cell centres:
    allocate(xc(0:nx), yc(0:ny), zc(0:nz))
    read(1) xc
    read(1) yc
    read(1) zc
    ! - magnetic field on cell faces:
    allocate(bx(0:nx-1,0:ny,0:nz), by(0:nx,0:ny-1,0:nz), bz(0:nx,0:ny,0:nz-1))
    read(1) bx
    read(1) by
    read(1) bz
    close(1)
    
    !-------------------------------------------------------------
    ! Precompute interpolation coefficients etc:
    xmin = (/x(0), xc(0), xc(0), xc(0), x(0), x(0)/)
    ymin = (/yc(0), y(0), yc(0), y(0), yc(0), y(0)/)
    zmin = (/zc(0), zc(0), z(0), z(0), z(0), zc(0)/)
    xmax = (/x(nx-1), xc(nx), xc(nx), x(nx), xc(nx-1), xc(nx-1)/)
    ymax = (/yc(ny), y(ny-1), yc(ny), yc(ny-1), y(ny), yc(ny-1)/)
    zmax = (/zc(nz), zc(nz), z(nz-1), zc(nz-1), zc(nz-1), z(nz)/)
    dx = x(1) - x(0)
    dy = y(1) - y(0)
    dz = z(1) - z(0)
    
    !-------------------------------------------------------------
    ! Read A from file:
    open(1, file=trim(afile), form='unformatted')
    ! - vector potential on cell edges:
    allocate(ax(0:nx,0:ny-1,0:nz-1), ay(0:nx-1,0:ny,0:nz-1), az(0:nx-1,0:ny-1,0:nz))
    read(1) ax
    read(1) ay
    read(1) az
    close(1)    
    
    !-------------------------------------------------------------
    ! Read field-line startpoints from file:
    open(1, file=x0file, form='unformatted')
    read(1) nl
    allocate(x0(1:nl,1:3))
    read(1) x0
    close(1)

    !-------------------------------------------------------------
    ! Loop over startpoints and trace field lines:

    ! - declare field-line helicity array:
    allocate(flh(nl))
    flh = 0d0
    
    ! Maximum allowed step-size:
    read(hmax,*) maxds
    maxds = maxds*min(dx, dy, dz)

   ! !$omp parallel private(dirn,ddirn,ds,nxt,k1,ds_dt,x2,k2,dx1,dx2,x1,xl,flh0,flh1)
  !  !$omp do
    do i=1,nl    
       
       ! - initialise variables:
       xl = 0d0
       xl(1) = xmin(1) - dx
       
       ! - trace backward then forward:
       do dirn=-1,1,2
          ddirn = dble(dirn)
          ds = maxds
          nxt = nmax/2
       
          xl = x0(i,:)
          flh0 = integrand(xl)  
          
          k1 = interpb(xl)
          ds_dt = dsqrt(sum(k1**2))
          if (ds_dt < minB) exit  ! stop if null reached
          ds_dt = ds_dt*ddirn
          k1 = k1/ds_dt
          
          do
                x2 = xl + ds*k1
            
                ! - if left domain, do Euler step then stop:
                if ((x2(1).lt.xmin(1)).or.(x2(1).gt.xmax(1)).or. &
                    (x2(2).lt.ymin(2)).or.(x2(2).gt.ymax(2)).or. &
                    (x2(3).lt.zmin(3)).or.(x2(3).gt.zmax(3))) then
                    if ((x2(1).lt.xmin(1)).or.(x2(1).gt.xmax(1))) then
                        if (k1(1).lt.0.0d0) then
                            ds1 = (xmin(1) - xl(1))/k1(1)
                        else
                            ds1 = (xmax(1) - xl(1))/k1(1)
                        end if
                    else
                        ds1 = 1.0d6
                    end if
                    if ((x2(2).lt.ymin(2)).or.(x2(2).gt.ymax(2))) then
                        if (k1(2).lt.0.0d0) then
                            ds2 = (ymin(2) - xl(2))/k1(2)
                        else
                            ds2 = (ymax(2) - xl(2))/k1(2)
                        end if
                    else
                        ds2 = 1.0d6                       
                    end if
                    if ((x2(3).lt.zmin(3)).or.(x2(3).gt.zmax(3))) then
                        if (k1(3).lt.0.0d0) then
                            ds3 = (zmin(3) - xl(3))/k1(3)
                        else
                            ds3 = (zmax(3) - xl(3))/k1(3)
                        end if
                    else
                        ds3 = 1.0d6                        
                    end if     
                    ds = min(ds1, ds2, ds3)
                    nxt = nxt + dirn
                    if ((nxt > nmax).or.(nxt < 1)) then
                        print*,'ERROR: field line overflow, increase nmax!'
                        stop
                    end if
                
                    xl = xl + ds*k1

                    flh1 = integrand(xl)
                    flh(i) = flh(i) + 0.5d0*ds*(flh0 + flh1)
                    flh0 = flh1
                    exit
                end if
                
                k2 = interpb(x2)
                ds_dt = dsqrt(sum(k2**2))
                if (ds_dt < minB) exit  ! stop if null reached
                ds_dt = ds_dt*ddirn
                k2 = k2/ds_dt

                dx1 = ds*k1
                dx2 = 0.5d0*ds*(k1 + k2)

                error = sum((dx2-dx1)**2)
                if (error.lt.1d-10) then
                    ds = maxds
                else
                    ds = min(maxds, 0.85*dabs(ds)*(maxerror/error)**0.25d0)
                end if

                if (error.le.maxerror) then

                    x1 = xl + dx2

                    ! - return midpoint if full step leaves domain:
                    if ((x1(1).lt.xmin(1)).or.(x1(1).gt.xmax(1)).or. &
                        (x1(2).lt.ymin(2)).or.(x1(2).gt.ymax(2)).or. &
                        (x1(3).lt.zmin(3)).or.(x1(3).gt.zmax(3))) x1 = x2
                    nxt = nxt + dirn
                    if ((nxt > nmax).or.(nxt < 1)) then
                        print*,'ERROR: field line overflow, increase nmax!'
                        stop
                    end if
                    
                    xl = x1
                    flh1 = integrand(xl)
                    flh(i) = flh(i) + 0.5d0*ds*(flh0 + flh1)
                    flh0 = flh1

                    k1 = interpb(xl)
                    ds_dt = dsqrt(sum(k1**2))
                    if (ds_dt < minB) exit  ! stop if null reached
                    ds_dt = ds_dt*ddirn
                    k1 = k1/ds_dt
                end if
            end do
        end do
    end do
   ! !$omp end do
   ! !$omp end parallel

    !-------------------------------------------------------------
    ! Output field lines to binary file:
    open(1, file=flhfile, form='unformatted')
    write(1) flh
    close(1)
    
contains
    
function interpb(xq)
    ! Return (bx, by, bz) at a specified point using linear interpolation.
    double precision, intent(in) :: xq(3)
    double precision :: interpb(3)
    double precision :: x1, y1, z1, fx, fy, fz
    integer :: ix, iy, iz
    
    ! bx
    x1 = (xq(1) - xmin(1))/dx
    y1 = (xq(2) - ymin(1))/dy
    z1 = (xq(3) - zmin(1))/dz
    ix = floor(x1)
    iy = floor(y1)
    iz = floor(z1)
    fx = x1 - dble(ix)
    fy = y1 - dble(iy)
    fz = z1 - dble(iz)
    interpb(1) = (1-fx)*(1-fy)*(1-fz)*bx(ix,iy,iz) + (1-fx)*(1-fy)*fz*bx(ix,iy,iz+1) &
        + (1-fx)*fy*(1-fz)*bx(ix,iy+1,iz) + (1-fx)*fy*fz*bx(ix,iy+1,iz+1) &
        + fx*(1-fy)*(1-fz)*bx(ix+1,iy,iz) + fx*(1-fy)*fz*bx(ix+1,iy,iz+1) &
        + fx*fy*(1-fz)*bx(ix+1,iy+1,iz) + fx*fy*fz*bx(ix+1,iy+1,iz+1)
    
    ! by
    x1 = (xq(1) - xmin(2))/dx
    y1 = (xq(2) - ymin(2))/dy
    z1 = (xq(3) - zmin(2))/dz
    ix = floor(x1)
    iy = floor(y1)
    iz = floor(z1)
    fx = x1 - dble(ix)
    fy = y1 - dble(iy)
    fz = z1 - dble(iz)
    interpb(2) = (1-fx)*(1-fy)*(1-fz)*by(ix,iy,iz) + (1-fx)*(1-fy)*fz*by(ix,iy,iz+1) &
        + (1-fx)*fy*(1-fz)*by(ix,iy+1,iz) + (1-fx)*fy*fz*by(ix,iy+1,iz+1) &
        + fx*(1-fy)*(1-fz)*by(ix+1,iy,iz) + fx*(1-fy)*fz*by(ix+1,iy,iz+1) &
        + fx*fy*(1-fz)*by(ix+1,iy+1,iz) + fx*fy*fz*by(ix+1,iy+1,iz+1)
        
    ! bz
    x1 = (xq(1) - xmin(3))/dx
    y1 = (xq(2) - ymin(3))/dy
    z1 = (xq(3) - zmin(3))/dz
    ix = floor(x1)
    iy = floor(y1)
    iz = floor(z1)
    fx = x1 - dble(ix)
    fy = y1 - dble(iy)
    fz = z1 - dble(iz)
    interpb(3) = (1-fx)*(1-fy)*(1-fz)*bz(ix,iy,iz) + (1-fx)*(1-fy)*fz*bz(ix,iy,iz+1) &
        + (1-fx)*fy*(1-fz)*bz(ix,iy+1,iz) + (1-fx)*fy*fz*bz(ix,iy+1,iz+1) &
        + fx*(1-fy)*(1-fz)*bz(ix+1,iy,iz) + fx*(1-fy)*fz*bz(ix+1,iy,iz+1) &
        + fx*fy*(1-fz)*bz(ix+1,iy+1,iz) + fx*fy*fz*bz(ix+1,iy+1,iz+1)
    
 end function interpb

 function interpa(xq)
    ! Return (ax, ay, az) at a specified point using linear interpolation.
    double precision, intent(in) :: xq(3)
    double precision :: interpa(3)
    double precision :: x1, y1, z1, fx, fy, fz
    integer :: ix, iy, iz
    
    ! ax
    x1 = (xq(1) - xmin(4))/dx
    y1 = (xq(2) - ymin(4))/dy
    z1 = (xq(3) - zmin(4))/dz
    ix = floor(x1)
    iy = floor(y1)
    iz = floor(z1)
    fx = x1 - dble(ix)
    fy = y1 - dble(iy)
    fz = z1 - dble(iz)
    interpa(1) = (1-fx)*(1-fy)*(1-fz)*ax(ix,iy,iz) + (1-fx)*(1-fy)*fz*ax(ix,iy,iz+1) &
        + (1-fx)*fy*(1-fz)*ax(ix,iy+1,iz) + (1-fx)*fy*fz*ax(ix,iy+1,iz+1) &
        + fx*(1-fy)*(1-fz)*ax(ix+1,iy,iz) + fx*(1-fy)*fz*ax(ix+1,iy,iz+1) &
        + fx*fy*(1-fz)*ax(ix+1,iy+1,iz) + fx*fy*fz*ax(ix+1,iy+1,iz+1)
    
    ! ay
    x1 = (xq(1) - xmin(5))/dx
    y1 = (xq(2) - ymin(5))/dy
    z1 = (xq(3) - zmin(5))/dz
    ix = floor(x1)
    iy = floor(y1)
    iz = floor(z1)
    fx = x1 - dble(ix)
    fy = y1 - dble(iy)
    fz = z1 - dble(iz)
    interpa(2) = (1-fx)*(1-fy)*(1-fz)*ay(ix,iy,iz) + (1-fx)*(1-fy)*fz*ay(ix,iy,iz+1) &
        + (1-fx)*fy*(1-fz)*ay(ix,iy+1,iz) + (1-fx)*fy*fz*ay(ix,iy+1,iz+1) &
        + fx*(1-fy)*(1-fz)*ay(ix+1,iy,iz) + fx*(1-fy)*fz*ay(ix+1,iy,iz+1) &
        + fx*fy*(1-fz)*ay(ix+1,iy+1,iz) + fx*fy*fz*ay(ix+1,iy+1,iz+1)
        
    ! az
    x1 = (xq(1) - xmin(6))/dx
    y1 = (xq(2) - ymin(6))/dy
    z1 = (xq(3) - zmin(6))/dz
    ix = floor(x1)
    iy = floor(y1)
    iz = floor(z1)
    fx = x1 - dble(ix)
    fy = y1 - dble(iy)
    fz = z1 - dble(iz)
    interpa(3) = (1-fx)*(1-fy)*(1-fz)*az(ix,iy,iz) + (1-fx)*(1-fy)*fz*az(ix,iy,iz+1) &
        + (1-fx)*fy*(1-fz)*az(ix,iy+1,iz) + (1-fx)*fy*fz*az(ix,iy+1,iz+1) &
        + fx*(1-fy)*(1-fz)*az(ix+1,iy,iz) + fx*(1-fy)*fz*az(ix+1,iy,iz+1) &
        + fx*fy*(1-fz)*az(ix+1,iy+1,iz) + fx*fy*fz*az(ix+1,iy+1,iz+1)
                
 end function interpa
 
 function integrand(xq)
    ! Return A.B/|B| at a specified point.
    double precision, intent(in) :: xq(3)
    double precision :: integrand, a(3), b(3)
 
    a = interpa(xq)
    b = interpb(xq)
    integrand = sum(a*b)/dsqrt(sum(b**2))
 
 end function integrand
 
end program fastflh



