add_library(imutils imutils.cpp)
target_link_libraries(imutils ${OpenCV_LIBS})

add_library(anti_shredder anti_shredder.cpp)
target_link_libraries(anti_shredder ${OpenCV_LIBS})

add_executable(main main.cpp)
target_link_libraries(main imutils anti_shredder ${OpenCV_LIBS})
