! FASTFL.F90 - cartesian field-line tracer using midpoint method
!            - uses openmp
!
! anthony.yeates@durham.ac.uk
!
!================================================================

!****************************************************************
program fastfl
!----------------------------------------------------------------
! Read in magnetic field datacube and field-line startpoints, 
! trace field lines, and output complete field lines.
!----------------------------------------------------------------

    implicit none
    
    character(128) :: bfile
    character*(*), parameter :: x0file='x0.unf', xlfile='xl.unf'
    integer :: nx, ny, nz, nl, i
    double precision, dimension(:), allocatable :: x, y, z, xc, yc, zc
    double precision, dimension(:,:), allocatable :: x0
    double precision, dimension(:,:,:), allocatable :: bx, by, bz, xl
    double precision, dimension(3) :: xmin, ymin, zmin, xmax, ymax, zmax
    double precision, dimension(3) :: k1, k2, dx1, dx2, x1, x2
    double precision :: dx, dy, dz, ddirn, maxds, ds, ds1, ds2, ds3, ds_dt, error
    integer :: dirn, nxt
    integer, parameter :: nmax=1000
    double precision, parameter :: minB=1d-4, maxerror=1d-1

    !-------------------------------------------------------------
    ! Get name of magnetic field from the command line:
    call get_command_argument(1, bfile)
    
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
    xmin = (/x(0), xc(0), xc(0)/)
    ymin = (/yc(0), y(0), yc(0)/)
    zmin = (/zc(0), zc(0), z(0)/)
    xmax = (/x(nx-1), xc(nx), xc(nx)/)
    ymax = (/yc(ny), y(ny-1), yc(ny)/)
    zmax = (/zc(nz), zc(nz), z(nz-1)/)
    dx = x(1) - x(0)
    dy = y(1) - y(0)
    dz = z(1) - z(0)
    
    !-------------------------------------------------------------
    ! Read field-line startpoints from file:
    open(1, file=x0file, form='unformatted')
    read(1) nl
    allocate(x0(1:nl,1:3))
    read(1) x0
    close(1)

    !-------------------------------------------------------------
    ! Loop over startpoints and trace field lines:

    ! - declare field line array:
    allocate(xl(nl,nmax,3))
    
    ! Maximum allowed step-size:
    maxds = dz
   
    !$omp parallel private(dirn,ddirn,ds,nxt,k1,ds_dt,x2,k2,dx1,dx2,x1)
    !$omp do
    do i=1,nl    
       
       ! - initialise variables:
       xl(i,:,:) = 0d0
       xl(i,:,1) = xmin(1) - dx
       
       ! - trace backward then forward:
       do dirn=-1,1,2
          ddirn = dble(dirn)
          ds = maxds
          nxt = nmax/2
       
          xl(i,nxt,:) = x0(i,:)

          k1 = interp1(xl(i,nxt,:))
          ds_dt = dsqrt(sum(k1**2))
          if (ds_dt < minB) exit  ! stop if null reached
          ds_dt = ds_dt*ddirn
          k1 = k1/ds_dt
          
            do
                x2 = xl(i,nxt,:) + ds*k1
            
                ! - if left domain, do Euler step then stop:
                if ((x2(1).lt.xmin(1)).or.(x2(1).gt.xmax(1)).or. &
                    (x2(2).lt.ymin(2)).or.(x2(2).gt.ymax(2)).or. &
                    (x2(3).lt.zmin(3)).or.(x2(3).gt.zmax(3))) then
                    if ((x2(1).lt.xmin(1)).or.(x2(1).gt.xmax(1))) then
                        if (k1(1).lt.0.0d0) then
                            ds1 = (xmin(1) - xl(i,nxt,1))/k1(1)
                        else
                            ds1 = (xmax(1) - xl(i,nxt,1))/k1(1)
                        end if
                    else
                        ds1 = 1.0d6
                    end if
                    if ((x2(2).lt.ymin(2)).or.(x2(2).gt.ymax(2))) then
                        if (k1(2).lt.0.0d0) then
                            ds2 = (ymin(2) - xl(i,nxt,2))/k1(2)
                        else
                            ds2 = (ymax(2) - xl(i,nxt,2))/k1(2)
                        end if
                    else
                        ds2 = 1.0d6                      
                    end if
                    if ((x2(3).lt.zmin(3)).or.(x2(3).gt.zmax(3))) then
                        if (k1(3).lt.0.0d0) then
                            ds3 = (zmin(3) - xl(i,nxt,3))/k1(3)
                        else
                            ds3 = (zmax(3) - xl(i,nxt,3))/k1(3)
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
                
                    xl(i,nxt,:) = xl(i,nxt-dirn,:) + ds*k1
                    exit
                end if
             
                k2 = interp1(x2)
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
                    x1 = xl(i,nxt,:) + dx2
                    ! - return midpoint if full step leaves domain:
                    if ((x1(1).lt.xmin(1)).or.(x1(1).gt.xmax(1)).or. &
                        (x1(2).lt.ymin(2)).or.(x1(2).gt.ymax(2)).or. &
                        (x1(3).lt.zmin(3)).or.(x1(3).gt.zmax(3))) x1 = x2
                    nxt = nxt + dirn
                    if ((nxt > nmax).or.(nxt < 1)) then
                        print*,'ERROR: field line overflow, increase nmax!'
                        stop
                    end if
                    
                    xl(i,nxt,:) = x1
                
                    k1 = interp1(xl(i,nxt,:))
                    ds_dt = dsqrt(sum(k1**2))
                    if (ds_dt < minB) exit  ! stop if null reached
                    ds_dt = ds_dt*ddirn
                    k1 = k1/ds_dt
                end if
            end do
        end do
    end do
    !$omp end do
    !$omp end parallel
       
    !-------------------------------------------------------------
    ! Output field lines to binary file:
    open(1, file=xlfile, form='unformatted')
    write(1) nmax
    write(1) xl
    close(1)
    
contains
    
function interp1(xq)
    ! Return (bx, by, bz) at a specified point using linear interpolation.
    double precision, intent(in) :: xq(3)
    double precision :: interp1(3)
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
    interp1(1) = (1-fx)*(1-fy)*(1-fz)*bx(ix,iy,iz) + (1-fx)*(1-fy)*fz*bx(ix,iy,iz+1) &
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
    interp1(2) = (1-fx)*(1-fy)*(1-fz)*by(ix,iy,iz) + (1-fx)*(1-fy)*fz*by(ix,iy,iz+1) &
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
    interp1(3) = (1-fx)*(1-fy)*(1-fz)*bz(ix,iy,iz) + (1-fx)*(1-fy)*fz*bz(ix,iy,iz+1) &
        + (1-fx)*fy*(1-fz)*bz(ix,iy+1,iz) + (1-fx)*fy*fz*bz(ix,iy+1,iz+1) &
        + fx*(1-fy)*(1-fz)*bz(ix+1,iy,iz) + fx*(1-fy)*fz*bz(ix+1,iy,iz+1) &
        + fx*fy*(1-fz)*bz(ix+1,iy+1,iz) + fx*fy*fz*bz(ix+1,iy+1,iz+1)
                
 end function interp1

end program fastfl



