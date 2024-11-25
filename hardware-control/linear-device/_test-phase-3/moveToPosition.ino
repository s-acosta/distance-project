bool moveToPosition(int desired_position){

  if (current_position != desired_position){

      if (millis()-last_ms > SLOW_MOTOR_PERIOD) { 
        moveSteps(1);
        
        last_ms = millis();

        if (forward_movement == true){
          current_position += 1;
        }
        else if (forward_movement == false){
          current_position -= 1;
        }
      }

    return false;
    }
    else{
      return true;
    }
}