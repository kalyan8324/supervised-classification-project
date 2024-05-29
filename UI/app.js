function login(){
    $username = $_POST['username'];
    $password = $_POST['password'];
    $password = md5($password);
    
}