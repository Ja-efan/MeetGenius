-- MySQL dump 10.13  Distrib 8.0.40, for Win64 (x86_64)
--
-- Host: localhost    Database: test_maria
-- ------------------------------------------------------
-- Server version	11.6.2-MariaDB

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8mb4 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `account_emailaddress`
--

DROP TABLE IF EXISTS `account_emailaddress`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `account_emailaddress` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `email` varchar(254) NOT NULL,
  `verified` tinyint(1) NOT NULL,
  `primary` tinyint(1) NOT NULL,
  `user_id` bigint(20) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `account_emailaddress_user_id_email_987c8728_uniq` (`user_id`,`email`),
  KEY `account_emailaddress_email_03be32b2` (`email`),
  CONSTRAINT `account_emailaddress_user_id_2c513194_fk_accounts_user_id` FOREIGN KEY (`user_id`) REFERENCES `accounts_user` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_uca1400_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `account_emailaddress`
--

LOCK TABLES `account_emailaddress` WRITE;
/*!40000 ALTER TABLE `account_emailaddress` DISABLE KEYS */;
/*!40000 ALTER TABLE `account_emailaddress` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `account_emailconfirmation`
--

DROP TABLE IF EXISTS `account_emailconfirmation`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `account_emailconfirmation` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `created` datetime(6) NOT NULL,
  `sent` datetime(6) DEFAULT NULL,
  `key` varchar(64) NOT NULL,
  `email_address_id` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `key` (`key`),
  KEY `account_emailconfirm_email_address_id_5b7f8c58_fk_account_e` (`email_address_id`),
  CONSTRAINT `account_emailconfirm_email_address_id_5b7f8c58_fk_account_e` FOREIGN KEY (`email_address_id`) REFERENCES `account_emailaddress` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_uca1400_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `account_emailconfirmation`
--

LOCK TABLES `account_emailconfirmation` WRITE;
/*!40000 ALTER TABLE `account_emailconfirmation` DISABLE KEYS */;
/*!40000 ALTER TABLE `account_emailconfirmation` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `accounts_department`
--

DROP TABLE IF EXISTS `accounts_department`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `accounts_department` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `name` varchar(50) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `name` (`name`)
) ENGINE=InnoDB AUTO_INCREMENT=3 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_uca1400_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `accounts_department`
--

LOCK TABLES `accounts_department` WRITE;
/*!40000 ALTER TABLE `accounts_department` DISABLE KEYS */;
INSERT INTO `accounts_department` VALUES (1,'개발'),(2,'마케팅');
/*!40000 ALTER TABLE `accounts_department` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `accounts_notification`
--

DROP TABLE IF EXISTS `accounts_notification`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `accounts_notification` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `message` longtext NOT NULL,
  `is_read` tinyint(1) NOT NULL,
  `created_at` datetime(6) NOT NULL,
  `user_id` bigint(20) NOT NULL,
  PRIMARY KEY (`id`),
  KEY `accounts_notification_user_id_30e6cfc5_fk_accounts_user_id` (`user_id`),
  CONSTRAINT `accounts_notification_user_id_30e6cfc5_fk_accounts_user_id` FOREIGN KEY (`user_id`) REFERENCES `accounts_user` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_uca1400_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `accounts_notification`
--

LOCK TABLES `accounts_notification` WRITE;
/*!40000 ALTER TABLE `accounts_notification` DISABLE KEYS */;
/*!40000 ALTER TABLE `accounts_notification` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `accounts_position`
--

DROP TABLE IF EXISTS `accounts_position`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `accounts_position` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `name` varchar(50) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `name` (`name`)
) ENGINE=InnoDB AUTO_INCREMENT=4 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_uca1400_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `accounts_position`
--

LOCK TABLES `accounts_position` WRITE;
/*!40000 ALTER TABLE `accounts_position` DISABLE KEYS */;
INSERT INTO `accounts_position` VALUES (3,'과장'),(2,'대리'),(1,'사원');
/*!40000 ALTER TABLE `accounts_position` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `accounts_user`
--

DROP TABLE IF EXISTS `accounts_user`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `accounts_user` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `password` varchar(128) NOT NULL,
  `last_login` datetime(6) DEFAULT NULL,
  `employee_number` varchar(7) NOT NULL,
  `name` varchar(50) NOT NULL,
  `email` varchar(254) NOT NULL,
  `is_active` tinyint(1) NOT NULL,
  `is_staff` tinyint(1) NOT NULL,
  `is_superuser` tinyint(1) NOT NULL,
  `department_id` bigint(20) DEFAULT NULL,
  `position_id` bigint(20) DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `employee_number` (`employee_number`),
  KEY `accounts_user_department_id_8dc06840_fk_accounts_department_id` (`department_id`),
  KEY `accounts_user_position_id_75cb83eb_fk_accounts_position_id` (`position_id`),
  CONSTRAINT `accounts_user_department_id_8dc06840_fk_accounts_department_id` FOREIGN KEY (`department_id`) REFERENCES `accounts_department` (`id`),
  CONSTRAINT `accounts_user_position_id_75cb83eb_fk_accounts_position_id` FOREIGN KEY (`position_id`) REFERENCES `accounts_position` (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=10 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_uca1400_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `accounts_user`
--

LOCK TABLES `accounts_user` WRITE;
/*!40000 ALTER TABLE `accounts_user` DISABLE KEYS */;
INSERT INTO `accounts_user` VALUES (1,'pbkdf2_sha256$870000$FcWV8qi88uBYouIiMjlHrt$TCIrbLKkDgVA9u9tyznYNNIuiDhc16/RZD2VqO1Kf8I=','2025-02-20 17:21:34.462153','1234567','','',1,1,1,NULL,NULL),(2,'pbkdf2_sha256$870000$JGNnGehMuQOfUt0uuiWK5G$dgMJY7LSiQNx8/e+Xlpd5Idt0QNEsV2TypPG+wZ6Jpo=','2025-02-20 19:27:48.746220','1230001','김도도','1@ssafy.com',1,0,0,1,1),(3,'pbkdf2_sha256$870000$QIs93vaH7NJKTVK9nvm72T$hnkbhipna7ubtyWhLIfii63oLsRb0OjVYxHdpUgS0bQ=','2025-02-20 17:40:24.834084','1230002','이레레','2@ssafy.com',1,0,0,1,2),(4,'pbkdf2_sha256$870000$00c6V8hMkqpEi8i0HR6r8q$VpSa1IPkcAmAPsJxkjs6hOgSqxJ9m3boE+5SMvjZMBo=',NULL,'1230003','박미미','3@ssafy.com',1,0,0,1,2),(5,'pbkdf2_sha256$870000$6HjuW7p6hC7CUg62ZgWFfT$ICMimVVF8WnwaYZK52oSzBiUZ2PhxkW8WWQ+lEcPhHQ=',NULL,'1230004','정파파','4@ssafy.com',1,0,0,1,3),(6,'pbkdf2_sha256$870000$I1dVCAddJR79zh2n9jxpe5$X2IIaEEhWJA+Uq9fM0O++h3t9kBZFNqKas0rJCZUJmY=',NULL,'1230005','장솔솔','5@ssafy.com',1,0,0,2,2),(7,'pbkdf2_sha256$870000$TWPiDvL8qCh0tBcOcqykqC$cevRLMt/6Zr6U+JbKkQAxvOWgv1nIy5bIpJg7MWNRgw=',NULL,'1230006','연라라','6@ssafy.com',1,0,0,2,2),(8,'pbkdf2_sha256$870000$2OZ4vXPNxeYU14UfIjpVZd$dgolWeYjz+1fbt8hkHBXWoXh/qHeUuV2Q9xBNCB9FBs=',NULL,'1230007','최시시','7@ssafy.com',1,0,0,2,3),(9,'pbkdf2_sha256$870000$V3bPtHCRPUJqDAirhnt3xv$BDlrDK/7sr19DgS2OYfJccbZ06Pjk/fdBiGjYACvnfA=',NULL,'1230008','김루루','9@ssafy.com',1,0,0,2,1);
/*!40000 ALTER TABLE `accounts_user` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `accounts_user_groups`
--

DROP TABLE IF EXISTS `accounts_user_groups`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `accounts_user_groups` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `user_id` bigint(20) NOT NULL,
  `group_id` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `accounts_user_groups_user_id_group_id_59c0b32f_uniq` (`user_id`,`group_id`),
  KEY `accounts_user_groups_group_id_bd11a704_fk_auth_group_id` (`group_id`),
  CONSTRAINT `accounts_user_groups_group_id_bd11a704_fk_auth_group_id` FOREIGN KEY (`group_id`) REFERENCES `auth_group` (`id`),
  CONSTRAINT `accounts_user_groups_user_id_52b62117_fk_accounts_user_id` FOREIGN KEY (`user_id`) REFERENCES `accounts_user` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_uca1400_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `accounts_user_groups`
--

LOCK TABLES `accounts_user_groups` WRITE;
/*!40000 ALTER TABLE `accounts_user_groups` DISABLE KEYS */;
/*!40000 ALTER TABLE `accounts_user_groups` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `accounts_user_user_permissions`
--

DROP TABLE IF EXISTS `accounts_user_user_permissions`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `accounts_user_user_permissions` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `user_id` bigint(20) NOT NULL,
  `permission_id` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `accounts_user_user_permi_user_id_permission_id_2ab516c2_uniq` (`user_id`,`permission_id`),
  KEY `accounts_user_user_p_permission_id_113bb443_fk_auth_perm` (`permission_id`),
  CONSTRAINT `accounts_user_user_p_permission_id_113bb443_fk_auth_perm` FOREIGN KEY (`permission_id`) REFERENCES `auth_permission` (`id`),
  CONSTRAINT `accounts_user_user_p_user_id_e4f0a161_fk_accounts_` FOREIGN KEY (`user_id`) REFERENCES `accounts_user` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_uca1400_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `accounts_user_user_permissions`
--

LOCK TABLES `accounts_user_user_permissions` WRITE;
/*!40000 ALTER TABLE `accounts_user_user_permissions` DISABLE KEYS */;
/*!40000 ALTER TABLE `accounts_user_user_permissions` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `auth_group`
--

DROP TABLE IF EXISTS `auth_group`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `auth_group` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `name` varchar(150) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `name` (`name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_uca1400_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `auth_group`
--

LOCK TABLES `auth_group` WRITE;
/*!40000 ALTER TABLE `auth_group` DISABLE KEYS */;
/*!40000 ALTER TABLE `auth_group` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `auth_group_permissions`
--

DROP TABLE IF EXISTS `auth_group_permissions`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `auth_group_permissions` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `group_id` int(11) NOT NULL,
  `permission_id` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `auth_group_permissions_group_id_permission_id_0cd325b0_uniq` (`group_id`,`permission_id`),
  KEY `auth_group_permissio_permission_id_84c5c92e_fk_auth_perm` (`permission_id`),
  CONSTRAINT `auth_group_permissio_permission_id_84c5c92e_fk_auth_perm` FOREIGN KEY (`permission_id`) REFERENCES `auth_permission` (`id`),
  CONSTRAINT `auth_group_permissions_group_id_b120cbf9_fk_auth_group_id` FOREIGN KEY (`group_id`) REFERENCES `auth_group` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_uca1400_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `auth_group_permissions`
--

LOCK TABLES `auth_group_permissions` WRITE;
/*!40000 ALTER TABLE `auth_group_permissions` DISABLE KEYS */;
/*!40000 ALTER TABLE `auth_group_permissions` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `auth_permission`
--

DROP TABLE IF EXISTS `auth_permission`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `auth_permission` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `name` varchar(255) NOT NULL,
  `content_type_id` int(11) NOT NULL,
  `codename` varchar(100) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `auth_permission_content_type_id_codename_01ab375a_uniq` (`content_type_id`,`codename`),
  CONSTRAINT `auth_permission_content_type_id_2f476e4b_fk_django_co` FOREIGN KEY (`content_type_id`) REFERENCES `django_content_type` (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=89 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_uca1400_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `auth_permission`
--

LOCK TABLES `auth_permission` WRITE;
/*!40000 ALTER TABLE `auth_permission` DISABLE KEYS */;
INSERT INTO `auth_permission` VALUES (1,'Can add department',1,'add_department'),(2,'Can change department',1,'change_department'),(3,'Can delete department',1,'delete_department'),(4,'Can view department',1,'view_department'),(5,'Can add position',2,'add_position'),(6,'Can change position',2,'change_position'),(7,'Can delete position',2,'delete_position'),(8,'Can view position',2,'view_position'),(9,'Can add user',3,'add_user'),(10,'Can change user',3,'change_user'),(11,'Can delete user',3,'delete_user'),(12,'Can view user',3,'view_user'),(13,'Can add notification',4,'add_notification'),(14,'Can change notification',4,'change_notification'),(15,'Can delete notification',4,'delete_notification'),(16,'Can view notification',4,'view_notification'),(17,'Can add project',5,'add_project'),(18,'Can change project',5,'change_project'),(19,'Can delete project',5,'delete_project'),(20,'Can view project',5,'view_project'),(21,'Can add document',6,'add_document'),(22,'Can change document',6,'change_document'),(23,'Can delete document',6,'delete_document'),(24,'Can view document',6,'view_document'),(25,'Can add report',7,'add_report'),(26,'Can change report',7,'change_report'),(27,'Can delete report',7,'delete_report'),(28,'Can view report',7,'view_report'),(29,'Can add project participation',8,'add_projectparticipation'),(30,'Can change project participation',8,'change_projectparticipation'),(31,'Can delete project participation',8,'delete_projectparticipation'),(32,'Can view project participation',8,'view_projectparticipation'),(33,'Can add meeting',9,'add_meeting'),(34,'Can change meeting',9,'change_meeting'),(35,'Can delete meeting',9,'delete_meeting'),(36,'Can view meeting',9,'view_meeting'),(37,'Can add agenda',10,'add_agenda'),(38,'Can change agenda',10,'change_agenda'),(39,'Can delete agenda',10,'delete_agenda'),(40,'Can view agenda',10,'view_agenda'),(41,'Can add mom',11,'add_mom'),(42,'Can change mom',11,'change_mom'),(43,'Can delete mom',11,'delete_mom'),(44,'Can view mom',11,'view_mom'),(45,'Can add summary mom',12,'add_summarymom'),(46,'Can change summary mom',12,'change_summarymom'),(47,'Can delete summary mom',12,'delete_summarymom'),(48,'Can view summary mom',12,'view_summarymom'),(49,'Can add meeting participation',13,'add_meetingparticipation'),(50,'Can change meeting participation',13,'change_meetingparticipation'),(51,'Can delete meeting participation',13,'delete_meetingparticipation'),(52,'Can view meeting participation',13,'view_meetingparticipation'),(53,'Can add Token',14,'add_token'),(54,'Can change Token',14,'change_token'),(55,'Can delete Token',14,'delete_token'),(56,'Can view Token',14,'view_token'),(57,'Can add Token',15,'add_tokenproxy'),(58,'Can change Token',15,'change_tokenproxy'),(59,'Can delete Token',15,'delete_tokenproxy'),(60,'Can view Token',15,'view_tokenproxy'),(61,'Can add log entry',16,'add_logentry'),(62,'Can change log entry',16,'change_logentry'),(63,'Can delete log entry',16,'delete_logentry'),(64,'Can view log entry',16,'view_logentry'),(65,'Can add permission',17,'add_permission'),(66,'Can change permission',17,'change_permission'),(67,'Can delete permission',17,'delete_permission'),(68,'Can view permission',17,'view_permission'),(69,'Can add group',18,'add_group'),(70,'Can change group',18,'change_group'),(71,'Can delete group',18,'delete_group'),(72,'Can view group',18,'view_group'),(73,'Can add content type',19,'add_contenttype'),(74,'Can change content type',19,'change_contenttype'),(75,'Can delete content type',19,'delete_contenttype'),(76,'Can view content type',19,'view_contenttype'),(77,'Can add session',20,'add_session'),(78,'Can change session',20,'change_session'),(79,'Can delete session',20,'delete_session'),(80,'Can view session',20,'view_session'),(81,'Can add email address',21,'add_emailaddress'),(82,'Can change email address',21,'change_emailaddress'),(83,'Can delete email address',21,'delete_emailaddress'),(84,'Can view email address',21,'view_emailaddress'),(85,'Can add email confirmation',22,'add_emailconfirmation'),(86,'Can change email confirmation',22,'change_emailconfirmation'),(87,'Can delete email confirmation',22,'delete_emailconfirmation'),(88,'Can view email confirmation',22,'view_emailconfirmation');
/*!40000 ALTER TABLE `auth_permission` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `authtoken_token`
--

DROP TABLE IF EXISTS `authtoken_token`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `authtoken_token` (
  `key` varchar(40) NOT NULL,
  `created` datetime(6) NOT NULL,
  `user_id` bigint(20) NOT NULL,
  PRIMARY KEY (`key`),
  UNIQUE KEY `user_id` (`user_id`),
  CONSTRAINT `authtoken_token_user_id_35299eff_fk_accounts_user_id` FOREIGN KEY (`user_id`) REFERENCES `accounts_user` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_uca1400_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `authtoken_token`
--

LOCK TABLES `authtoken_token` WRITE;
/*!40000 ALTER TABLE `authtoken_token` DISABLE KEYS */;
INSERT INTO `authtoken_token` VALUES ('84de3e20c63cfea6631577f62258596c9811b65e','2025-02-20 17:40:23.458477',3),('931d1c2b94b8feb9d17d2319bd65f1a153eee5d6','2025-02-20 17:40:14.549561',2);
/*!40000 ALTER TABLE `authtoken_token` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `django_admin_log`
--

DROP TABLE IF EXISTS `django_admin_log`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `django_admin_log` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `action_time` datetime(6) NOT NULL,
  `object_id` longtext DEFAULT NULL,
  `object_repr` varchar(200) NOT NULL,
  `action_flag` smallint(5) unsigned NOT NULL CHECK (`action_flag` >= 0),
  `change_message` longtext NOT NULL,
  `content_type_id` int(11) DEFAULT NULL,
  `user_id` bigint(20) NOT NULL,
  PRIMARY KEY (`id`),
  KEY `django_admin_log_content_type_id_c4bce8eb_fk_django_co` (`content_type_id`),
  KEY `django_admin_log_user_id_c564eba6_fk_accounts_user_id` (`user_id`),
  CONSTRAINT `django_admin_log_content_type_id_c4bce8eb_fk_django_co` FOREIGN KEY (`content_type_id`) REFERENCES `django_content_type` (`id`),
  CONSTRAINT `django_admin_log_user_id_c564eba6_fk_accounts_user_id` FOREIGN KEY (`user_id`) REFERENCES `accounts_user` (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=14 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_uca1400_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `django_admin_log`
--

LOCK TABLES `django_admin_log` WRITE;
/*!40000 ALTER TABLE `django_admin_log` DISABLE KEYS */;
INSERT INTO `django_admin_log` VALUES (1,'2025-02-20 17:22:11.226522','1','개발',1,'[{\"added\": {}}]',1,1),(2,'2025-02-20 17:22:14.886171','2','마케팅',1,'[{\"added\": {}}]',1,1),(3,'2025-02-20 17:22:23.879852','1','사원',1,'[{\"added\": {}}]',2,1),(4,'2025-02-20 17:22:26.782343','2','대리',1,'[{\"added\": {}}]',2,1),(5,'2025-02-20 17:22:29.897334','3','과장',1,'[{\"added\": {}}]',2,1),(6,'2025-02-20 17:22:53.589699','2','김도도',1,'[{\"added\": {}}]',3,1),(7,'2025-02-20 17:23:17.668730','3','이레레',1,'[{\"added\": {}}]',3,1),(8,'2025-02-20 17:23:37.233727','4','박미미',1,'[{\"added\": {}}]',3,1),(9,'2025-02-20 17:23:55.256154','5','정파파',1,'[{\"added\": {}}]',3,1),(10,'2025-02-20 17:24:14.150551','6','장솔솔',1,'[{\"added\": {}}]',3,1),(11,'2025-02-20 17:25:01.006535','7','연라라',1,'[{\"added\": {}}]',3,1),(12,'2025-02-20 17:25:17.716641','8','최시시',1,'[{\"added\": {}}]',3,1),(13,'2025-02-20 19:26:58.906479','9','김루루',1,'[{\"added\": {}}]',3,1);
/*!40000 ALTER TABLE `django_admin_log` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `django_content_type`
--

DROP TABLE IF EXISTS `django_content_type`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `django_content_type` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `app_label` varchar(100) NOT NULL,
  `model` varchar(100) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `django_content_type_app_label_model_76bd3d3b_uniq` (`app_label`,`model`)
) ENGINE=InnoDB AUTO_INCREMENT=23 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_uca1400_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `django_content_type`
--

LOCK TABLES `django_content_type` WRITE;
/*!40000 ALTER TABLE `django_content_type` DISABLE KEYS */;
INSERT INTO `django_content_type` VALUES (21,'account','emailaddress'),(22,'account','emailconfirmation'),(1,'accounts','department'),(4,'accounts','notification'),(2,'accounts','position'),(3,'accounts','user'),(16,'admin','logentry'),(18,'auth','group'),(17,'auth','permission'),(14,'authtoken','token'),(15,'authtoken','tokenproxy'),(19,'contenttypes','contenttype'),(10,'meetingroom','agenda'),(9,'meetingroom','meeting'),(13,'meetingroom','meetingparticipation'),(11,'meetingroom','mom'),(12,'meetingroom','summarymom'),(6,'projects','document'),(5,'projects','project'),(8,'projects','projectparticipation'),(7,'projects','report'),(20,'sessions','session');
/*!40000 ALTER TABLE `django_content_type` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `django_migrations`
--

DROP TABLE IF EXISTS `django_migrations`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `django_migrations` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `app` varchar(255) NOT NULL,
  `name` varchar(255) NOT NULL,
  `applied` datetime(6) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=35 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_uca1400_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `django_migrations`
--

LOCK TABLES `django_migrations` WRITE;
/*!40000 ALTER TABLE `django_migrations` DISABLE KEYS */;
INSERT INTO `django_migrations` VALUES (1,'contenttypes','0001_initial','2025-02-20 17:20:57.128027'),(2,'contenttypes','0002_remove_content_type_name','2025-02-20 17:20:57.172667'),(3,'auth','0001_initial','2025-02-20 17:20:57.333561'),(4,'auth','0002_alter_permission_name_max_length','2025-02-20 17:20:57.363783'),(5,'auth','0003_alter_user_email_max_length','2025-02-20 17:20:57.373615'),(6,'auth','0004_alter_user_username_opts','2025-02-20 17:20:57.380880'),(7,'auth','0005_alter_user_last_login_null','2025-02-20 17:20:57.388950'),(8,'auth','0006_require_contenttypes_0002','2025-02-20 17:20:57.388950'),(9,'auth','0007_alter_validators_add_error_messages','2025-02-20 17:20:57.393304'),(10,'auth','0008_alter_user_username_max_length','2025-02-20 17:20:57.397270'),(11,'auth','0009_alter_user_last_name_max_length','2025-02-20 17:20:57.406067'),(12,'auth','0010_alter_group_name_max_length','2025-02-20 17:20:57.416707'),(13,'auth','0011_update_proxy_permissions','2025-02-20 17:20:57.424962'),(14,'auth','0012_alter_user_first_name_max_length','2025-02-20 17:20:57.433055'),(15,'accounts','0001_initial','2025-02-20 17:20:57.720448'),(16,'account','0001_initial','2025-02-20 17:20:57.804127'),(17,'account','0002_email_max_length','2025-02-20 17:20:57.828086'),(18,'account','0003_alter_emailaddress_create_unique_verified_email','2025-02-20 17:20:57.879096'),(19,'account','0004_alter_emailaddress_drop_unique_email','2025-02-20 17:20:58.246199'),(20,'account','0005_emailaddress_idx_upper_email','2025-02-20 17:20:58.256968'),(21,'account','0006_emailaddress_lower','2025-02-20 17:20:58.279293'),(22,'account','0007_emailaddress_idx_email','2025-02-20 17:20:58.311369'),(23,'account','0008_emailaddress_unique_primary_email_fixup','2025-02-20 17:20:58.329197'),(24,'account','0009_emailaddress_unique_primary_email','2025-02-20 17:20:58.345035'),(25,'admin','0001_initial','2025-02-20 17:20:58.435464'),(26,'admin','0002_logentry_remove_auto_add','2025-02-20 17:20:58.444026'),(27,'admin','0003_logentry_add_action_flag_choices','2025-02-20 17:20:58.463805'),(28,'authtoken','0001_initial','2025-02-20 17:20:58.511553'),(29,'authtoken','0002_auto_20160226_1747','2025-02-20 17:20:58.596090'),(30,'authtoken','0003_tokenproxy','2025-02-20 17:20:58.606036'),(31,'authtoken','0004_alter_tokenproxy_options','2025-02-20 17:20:58.612494'),(32,'projects','0001_initial','2025-02-20 17:20:59.033791'),(33,'meetingroom','0001_initial','2025-02-20 17:20:59.522578'),(34,'sessions','0001_initial','2025-02-20 17:20:59.567238');
/*!40000 ALTER TABLE `django_migrations` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `django_session`
--

DROP TABLE IF EXISTS `django_session`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `django_session` (
  `session_key` varchar(40) NOT NULL,
  `session_data` longtext NOT NULL,
  `expire_date` datetime(6) NOT NULL,
  PRIMARY KEY (`session_key`),
  KEY `django_session_expire_date_a5c62663` (`expire_date`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_uca1400_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `django_session`
--

LOCK TABLES `django_session` WRITE;
/*!40000 ALTER TABLE `django_session` DISABLE KEYS */;
INSERT INTO `django_session` VALUES ('1lwub28078ox4vc818jhku8r1up3m22f','.eJxVjMEOwiAQBf-FsyHAQgGP3v0GsiwgVQNJaU_Gf9cmPej1zcx7sYDbWsM28hLmxM5MsdPvFpEeue0g3bHdOqfe1mWOfFf4QQe_9pSfl8P9O6g46rcWlkipiUqmpClaraUHScnbCF4IFICuZCgJDRowUVqMWFxG0HZySrL3B_loOE8:1tl3mW:dq5v6hAy2r3OGMBq7w0vG6AqyNJh9JU_q05UeJCde_Q','2025-03-06 19:27:48.753160'),('8ep1e0mrz9p1ap77e1b4h3s4gi4lerg7','.eJxVjDsOwjAQRO_iGlnxf01JzxmstXfBAeRIcVIh7k4ipYBqpHlv5i0SrktNa-c5jSTOQonTb5exPLntgB7Y7pMsU1vmMctdkQft8joRvy6H-3dQsddtnR1zLMWpEAfHoEvUSkWmAbLS6AtkDbRltGhsAOPQm2A9G2vJwA3E5wvdjjdW:1tl1oM:JzijuowXNQW1H7n3TuHQjyTTSCoUE2WP6ZOjBX4zqAE','2025-03-06 17:21:34.462153'),('90h49txenr357vnphsho90ai32lt82ml','.eJxVjEEOwiAQRe_C2hAClCku3XsGMjOMUjWQlHZlvLtt0oVu33v_v1XCdSlp7TKnKauzcur0ywj5KXUX-YH13jS3uswT6T3Rh-362rK8Lkf7d1Cwl20N1hmH2ZNA5shgPAe03riB4ggAFtgHC4PNAhxN9HJzGxUyBmKgUX2-yY429w:1tl26a:Y2Nsymlfoy1L3q7m9CgVrh2dg7fp9wE9czCGbQO9M50','2025-03-06 17:40:24.850302'),('awu6ehl6ty6htaqd0nwgjylnv6jgs3ex','.eJxVjEEOwiAQRe_C2hAClCku3XsGMjOMUjWQlHZlvLtt0oVu33v_v1XCdSlp7TKnKauzcur0ywj5KXUX-YH13jS3uswT6T3Rh-362rK8Lkf7d1Cwl20N1hmH2ZNA5shgPAe03riB4ggAFtgHC4PNAhxN9HJzGxUyBmKgUX2-yY429w:1tl26Z:XPB1DOAYOdneP4RN2esLhnqnUZQedCzc6l_5mrINiIk','2025-03-06 17:40:23.475012'),('tijll0wfly1j2yi5oq6xlf7lto3ruxxt','e30:1tl26Q:xe3Te10e3hdxSibXgFdH0KoFUwFXgaR1Xl3nzr3aCHI','2025-03-06 17:40:14.523501');
/*!40000 ALTER TABLE `django_session` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `meetingroom_agenda`
--

DROP TABLE IF EXISTS `meetingroom_agenda`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `meetingroom_agenda` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `order` int(11) NOT NULL,
  `title` varchar(100) NOT NULL,
  `meeting_id` bigint(20) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `meetingroom_agenda_meeting_id_3dc8efd7_fk_meetingroom_meeting_id` (`meeting_id`),
  CONSTRAINT `meetingroom_agenda_meeting_id_3dc8efd7_fk_meetingroom_meeting_id` FOREIGN KEY (`meeting_id`) REFERENCES `meetingroom_meeting` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_uca1400_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `meetingroom_agenda`
--

LOCK TABLES `meetingroom_agenda` WRITE;
/*!40000 ALTER TABLE `meetingroom_agenda` DISABLE KEYS */;
/*!40000 ALTER TABLE `meetingroom_agenda` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `meetingroom_meeting`
--

DROP TABLE IF EXISTS `meetingroom_meeting`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `meetingroom_meeting` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `room` int(11) NOT NULL,
  `starttime` datetime(6) NOT NULL,
  `endtime` datetime(6) NOT NULL,
  `booked_at` datetime(6) NOT NULL,
  `title` varchar(100) NOT NULL,
  `booker_id` bigint(20) DEFAULT NULL,
  `project_id` bigint(20) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `meetingroom_meeting_booker_id_78b5649a_fk_accounts_user_id` (`booker_id`),
  KEY `meetingroom_meeting_project_id_a319d712_fk_projects_project_id` (`project_id`),
  CONSTRAINT `meetingroom_meeting_booker_id_78b5649a_fk_accounts_user_id` FOREIGN KEY (`booker_id`) REFERENCES `accounts_user` (`id`),
  CONSTRAINT `meetingroom_meeting_project_id_a319d712_fk_projects_project_id` FOREIGN KEY (`project_id`) REFERENCES `projects_project` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_uca1400_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `meetingroom_meeting`
--

LOCK TABLES `meetingroom_meeting` WRITE;
/*!40000 ALTER TABLE `meetingroom_meeting` DISABLE KEYS */;
/*!40000 ALTER TABLE `meetingroom_meeting` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `meetingroom_meetingparticipation`
--

DROP TABLE IF EXISTS `meetingroom_meetingparticipation`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `meetingroom_meetingparticipation` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `authority` int(11) NOT NULL,
  `meeting_id` bigint(20) NOT NULL,
  `participant_id` bigint(20) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `meetingroom_meetingparti_meeting_id_participant_i_31849de3_uniq` (`meeting_id`,`participant_id`),
  KEY `meetingroom_meetingp_participant_id_f88e9c64_fk_accounts_` (`participant_id`),
  CONSTRAINT `meetingroom_meetingp_meeting_id_db17461e_fk_meetingro` FOREIGN KEY (`meeting_id`) REFERENCES `meetingroom_meeting` (`id`),
  CONSTRAINT `meetingroom_meetingp_participant_id_f88e9c64_fk_accounts_` FOREIGN KEY (`participant_id`) REFERENCES `accounts_user` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_uca1400_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `meetingroom_meetingparticipation`
--

LOCK TABLES `meetingroom_meetingparticipation` WRITE;
/*!40000 ALTER TABLE `meetingroom_meetingparticipation` DISABLE KEYS */;
/*!40000 ALTER TABLE `meetingroom_meetingparticipation` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `meetingroom_mom`
--

DROP TABLE IF EXISTS `meetingroom_mom`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `meetingroom_mom` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `agenda_result` longtext NOT NULL,
  `completed` tinyint(1) NOT NULL,
  `agenda_id` bigint(20) NOT NULL,
  `document_id` bigint(20) DEFAULT NULL,
  `meeting_id` bigint(20) NOT NULL,
  PRIMARY KEY (`id`),
  KEY `meetingroom_mom_agenda_id_a4227c1e_fk_meetingroom_agenda_id` (`agenda_id`),
  KEY `meetingroom_mom_document_id_3924b1a8_fk_projects_document_id` (`document_id`),
  KEY `meetingroom_mom_meeting_id_c30fcf5c_fk_meetingroom_meeting_id` (`meeting_id`),
  CONSTRAINT `meetingroom_mom_agenda_id_a4227c1e_fk_meetingroom_agenda_id` FOREIGN KEY (`agenda_id`) REFERENCES `meetingroom_agenda` (`id`),
  CONSTRAINT `meetingroom_mom_document_id_3924b1a8_fk_projects_document_id` FOREIGN KEY (`document_id`) REFERENCES `projects_document` (`id`),
  CONSTRAINT `meetingroom_mom_meeting_id_c30fcf5c_fk_meetingroom_meeting_id` FOREIGN KEY (`meeting_id`) REFERENCES `meetingroom_meeting` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_uca1400_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `meetingroom_mom`
--

LOCK TABLES `meetingroom_mom` WRITE;
/*!40000 ALTER TABLE `meetingroom_mom` DISABLE KEYS */;
/*!40000 ALTER TABLE `meetingroom_mom` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `meetingroom_summarymom`
--

DROP TABLE IF EXISTS `meetingroom_summarymom`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `meetingroom_summarymom` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `summary_result` longtext NOT NULL,
  `completed` tinyint(1) NOT NULL,
  `document_id` bigint(20) DEFAULT NULL,
  `mom_id` bigint(20) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `mom_id` (`mom_id`),
  KEY `meetingroom_summarym_document_id_28b61e44_fk_projects_` (`document_id`),
  CONSTRAINT `meetingroom_summarym_document_id_28b61e44_fk_projects_` FOREIGN KEY (`document_id`) REFERENCES `projects_document` (`id`),
  CONSTRAINT `meetingroom_summarymom_mom_id_2de2d621_fk_meetingroom_mom_id` FOREIGN KEY (`mom_id`) REFERENCES `meetingroom_mom` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_uca1400_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `meetingroom_summarymom`
--

LOCK TABLES `meetingroom_summarymom` WRITE;
/*!40000 ALTER TABLE `meetingroom_summarymom` DISABLE KEYS */;
/*!40000 ALTER TABLE `meetingroom_summarymom` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `projects_document`
--

DROP TABLE IF EXISTS `projects_document`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `projects_document` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `type` int(11) NOT NULL,
  `embedding` tinyint(1) NOT NULL,
  `department_id` bigint(20) NOT NULL,
  `project_id` bigint(20) NOT NULL,
  PRIMARY KEY (`id`),
  KEY `projects_document_department_id_10917348_fk_accounts_` (`department_id`),
  KEY `projects_document_project_id_7b4bc7df_fk_projects_project_id` (`project_id`),
  CONSTRAINT `projects_document_department_id_10917348_fk_accounts_` FOREIGN KEY (`department_id`) REFERENCES `accounts_department` (`id`),
  CONSTRAINT `projects_document_project_id_7b4bc7df_fk_projects_project_id` FOREIGN KEY (`project_id`) REFERENCES `projects_project` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_uca1400_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `projects_document`
--

LOCK TABLES `projects_document` WRITE;
/*!40000 ALTER TABLE `projects_document` DISABLE KEYS */;
/*!40000 ALTER TABLE `projects_document` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `projects_project`
--

DROP TABLE IF EXISTS `projects_project`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `projects_project` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `name` varchar(50) NOT NULL,
  `description` longtext NOT NULL,
  `is_inprogress` tinyint(1) NOT NULL,
  `startdate` datetime(6) NOT NULL,
  `duedate` datetime(6) NOT NULL,
  `creator_id` bigint(20) DEFAULT NULL,
  `department_id` bigint(20) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `name` (`name`),
  KEY `projects_project_creator_id_ef53b054_fk_accounts_user_id` (`creator_id`),
  KEY `projects_project_department_id_2d73b0cf_fk_accounts_` (`department_id`),
  CONSTRAINT `projects_project_creator_id_ef53b054_fk_accounts_user_id` FOREIGN KEY (`creator_id`) REFERENCES `accounts_user` (`id`),
  CONSTRAINT `projects_project_department_id_2d73b0cf_fk_accounts_` FOREIGN KEY (`department_id`) REFERENCES `accounts_department` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_uca1400_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `projects_project`
--

LOCK TABLES `projects_project` WRITE;
/*!40000 ALTER TABLE `projects_project` DISABLE KEYS */;
/*!40000 ALTER TABLE `projects_project` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `projects_projectparticipation`
--

DROP TABLE IF EXISTS `projects_projectparticipation`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `projects_projectparticipation` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `authority` int(11) NOT NULL,
  `participant_id` bigint(20) NOT NULL,
  `project_id` bigint(20) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `unique_project_participant` (`project_id`,`participant_id`),
  KEY `projects_projectpart_participant_id_743624b9_fk_accounts_` (`participant_id`),
  CONSTRAINT `projects_projectpart_participant_id_743624b9_fk_accounts_` FOREIGN KEY (`participant_id`) REFERENCES `accounts_user` (`id`),
  CONSTRAINT `projects_projectpart_project_id_eebd5e22_fk_projects_` FOREIGN KEY (`project_id`) REFERENCES `projects_project` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_uca1400_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `projects_projectparticipation`
--

LOCK TABLES `projects_projectparticipation` WRITE;
/*!40000 ALTER TABLE `projects_projectparticipation` DISABLE KEYS */;
/*!40000 ALTER TABLE `projects_projectparticipation` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `projects_report`
--

DROP TABLE IF EXISTS `projects_report`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `projects_report` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `title` varchar(100) NOT NULL,
  `content` longtext NOT NULL,
  `created_at` datetime(6) NOT NULL,
  `document_id` bigint(20) DEFAULT NULL,
  `project_id` bigint(20) NOT NULL,
  `writer_id` bigint(20) NOT NULL,
  PRIMARY KEY (`id`),
  KEY `projects_report_document_id_0e5f551b_fk_projects_document_id` (`document_id`),
  KEY `projects_report_project_id_2268fd1a_fk_projects_project_id` (`project_id`),
  KEY `projects_report_writer_id_ecb787aa_fk_accounts_user_id` (`writer_id`),
  CONSTRAINT `projects_report_document_id_0e5f551b_fk_projects_document_id` FOREIGN KEY (`document_id`) REFERENCES `projects_document` (`id`),
  CONSTRAINT `projects_report_project_id_2268fd1a_fk_projects_project_id` FOREIGN KEY (`project_id`) REFERENCES `projects_project` (`id`),
  CONSTRAINT `projects_report_writer_id_ecb787aa_fk_accounts_user_id` FOREIGN KEY (`writer_id`) REFERENCES `accounts_user` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_uca1400_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `projects_report`
--

LOCK TABLES `projects_report` WRITE;
/*!40000 ALTER TABLE `projects_report` DISABLE KEYS */;
/*!40000 ALTER TABLE `projects_report` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Dumping routines for database 'test_maria'
--
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2025-02-20 19:29:08

INSERT INTO projects_project 
  (name, description, is_inprogress, startdate, duedate, creator_id, department_id)
VALUES
  (
    'AI 기반 맞춤형 건강 관리 시스템',
    'AI 기반 맞춤형 건강 관리 시스템 개발',
    1,
    '2025-02-14 00:00:00',
    '2025-04-11 23:59:59',
    (SELECT id FROM accounts_user WHERE name = '정파파'),
    (SELECT id FROM accounts_department WHERE name = '개발')
  );
-- 저장된 새 프로젝트의 id를 변수에 할당
SET @project_id = LAST_INSERT_ID();

--------------------------------------------------------------------------------
-- 2. 프로젝트 참여자 등록
--------------------------------------------------------------------------------
INSERT INTO projects_projectparticipation (authority, participant_id, project_id)
VALUES
  (1, (SELECT id FROM accounts_user WHERE name = '정파파'), @project_id),
  (0, (SELECT id FROM accounts_user WHERE name = '박미미'), @project_id),
  (1, (SELECT id FROM accounts_user WHERE name = '최시시'), @project_id),
  (0, (SELECT id FROM accounts_user WHERE name = '장솔솔'), @project_id);

--------------------------------------------------------------------------------
-- 3. 회의 데이터 및 관련 정보 입력
--------------------------------------------------------------------------------
-- (a) 회의 데이터 삽입 (5회; 날짜: 2025-02-17 ~ 2025-02-21, 시간: 09:00 ~ 11:00)
SET @booker = (SELECT id FROM accounts_user WHERE name = '정파파');

-- Meeting 1: 2025-02-17 09:00 ~ 11:00, 제목: '기술적 요구 사항 정의'
INSERT INTO meetingroom_meeting 
  (room, starttime, endtime, booked_at, title, booker_id, project_id)
VALUES
  (1, '2025-02-17 09:00:00', '2025-02-17 11:00:00', '2025-02-17 09:00:00', '기술적 요구 사항 정의', @booker, @project_id);
SET @m1_new = LAST_INSERT_ID();

-- Meeting 2: 2025-02-18 09:00 ~ 11:00, 제목: '시스템 설계 및 개발'
INSERT INTO meetingroom_meeting 
  (room, starttime, endtime, booked_at, title, booker_id, project_id)
VALUES
  (1, '2025-02-18 09:00:00', '2025-02-18 11:00:00', '2025-02-18 09:00:00', '시스템 설계 및 개발', @booker, @project_id);
SET @m2_new = LAST_INSERT_ID();

-- Meeting 3: 2025-02-19 09:00 ~ 11:00, 제목: '마케팅 및 출시 전략'
INSERT INTO meetingroom_meeting 
  (room, starttime, endtime, booked_at, title, booker_id, project_id)
VALUES
  (1, '2025-02-19 09:00:00', '2025-02-19 11:00:00', '2025-02-19 09:00:00', '마케팅 및 출시 전략', @booker, @project_id);
SET @m3_new = LAST_INSERT_ID();

-- Meeting 4: 2025-02-20 09:00 ~ 11:00, 제목: '비즈니스 모델 및 수익화 전략 회의'
INSERT INTO meetingroom_meeting 
  (room, starttime, endtime, booked_at, title, booker_id, project_id)
VALUES
  (1, '2025-02-20 09:00:00', '2025-02-20 11:00:00', '2025-02-20 09:00:00', '비즈니스 모델 및 수익화 전략 회의', @booker, @project_id);
SET @m4_new = LAST_INSERT_ID();

-- Meeting 5: 2025-02-21 09:00 ~ 11:00, 제목: '개발 일정 조정 회의'
INSERT INTO meetingroom_meeting 
  (room, starttime, endtime, booked_at, title, booker_id, project_id)
VALUES
  (1, '2025-02-21 09:00:00', '2025-02-21 11:00:00', '2025-02-21 09:00:00', '개발 일정 조정 회의', @booker, @project_id);
SET @m5_new = LAST_INSERT_ID();

--------------------------------------------------------------------------------
-- (b) 각 회의별 안건(Agenda) 삽입
--------------------------------------------------------------------------------
/* Meeting 1: 세 개의 안건 */
INSERT INTO meetingroom_agenda (`order`, title, meeting_id) 
VALUES 
  (1, '프로젝트 관련 최신 이슈', @m1_new),
  (2, '데이터 파이프라인 검토', @m1_new),
  (3, '개발 일정 계획', @m1_new);
SET @a1_new = (SELECT id FROM meetingroom_agenda WHERE meeting_id=@m1_new AND title='프로젝트 관련 최신 이슈' ORDER BY id DESC LIMIT 1);
SET @a2_new = (SELECT id FROM meetingroom_agenda WHERE meeting_id=@m1_new AND title='데이터 파이프라인 검토' ORDER BY id DESC LIMIT 1);
SET @a3_new = (SELECT id FROM meetingroom_agenda WHERE meeting_id=@m1_new AND title='개발 일정 계획' ORDER BY id DESC LIMIT 1);

/* Meeting 2: 세 개의 안건 */
INSERT INTO meetingroom_agenda (`order`, title, meeting_id)
VALUES 
  (1, '개발 일정 수정', @m2_new),
  (2, '추천 알고리즘 및 AI 모델 결정', @m2_new),
  (3, '모의 테스트 결과 및 추후 테스트 계획 결정', @m2_new);
SET @a4_new = (SELECT id FROM meetingroom_agenda WHERE meeting_id=@m2_new AND title='개발 일정 수정' ORDER BY id DESC LIMIT 1);
SET @a5_new = (SELECT id FROM meetingroom_agenda WHERE meeting_id=@m2_new AND title='추천 알고리즘 및 AI 모델 결정' ORDER BY id DESC LIMIT 1);
SET @a6_new = (SELECT id FROM meetingroom_agenda WHERE meeting_id=@m2_new AND title='모의 테스트 결과 및 추후 테스트 계획 결정' ORDER BY id DESC LIMIT 1);

/* Meeting 3: 세 개의 안건 */
INSERT INTO meetingroom_agenda (`order`, title, meeting_id)
VALUES 
  (1, '제품 마케팅 전략 수립', @m3_new),
  (2, '브랜드 인지도 향상 방안', @m3_new),
  (3, '가격 전략 및 경쟁 분석', @m3_new);
SET @a7_new = (SELECT id FROM meetingroom_agenda WHERE meeting_id=@m3_new AND title='제품 마케팅 전략 수립' ORDER BY id DESC LIMIT 1);
SET @a8_new = (SELECT id FROM meetingroom_agenda WHERE meeting_id=@m3_new AND title='브랜드 인지도 향상 방안' ORDER BY id DESC LIMIT 1);
SET @a9_new = (SELECT id FROM meetingroom_agenda WHERE meeting_id=@m3_new AND title='가격 전략 및 경쟁 분석' ORDER BY id DESC LIMIT 1);

/* Meeting 4: 두 개의 안건 */
INSERT INTO meetingroom_agenda (`order`, title, meeting_id)
VALUES 
  (1, '수익화 모델 정의', @m4_new),
  (2, '가격 전략 및 경쟁 분석', @m4_new);
SET @a10_new = (SELECT id FROM meetingroom_agenda WHERE meeting_id=@m4_new AND title='수익화 모델 정의' ORDER BY id DESC LIMIT 1);
SET @a11_new = (SELECT id FROM meetingroom_agenda WHERE meeting_id=@m4_new AND title='가격 전략 및 경쟁 분석' ORDER BY id DESC LIMIT 1);

/* Meeting 5: 네 개의 안건 */
INSERT INTO meetingroom_agenda (`order`, title, meeting_id)
VALUES 
  (1, '프로젝트 진행 상황 점검', @m5_new),
  (2, '리스크 관리 및 문제 해결', @m5_new),
  (3, '고객 피드백 및 개선 계획', @m5_new),
  (4, '프로젝트 일정 조정 및 리소스 배분', @m5_new);
SET @a12_new = (SELECT id FROM meetingroom_agenda WHERE meeting_id=@m5_new AND title='프로젝트 진행 상황 점검' ORDER BY id DESC LIMIT 1);
SET @a13_new = (SELECT id FROM meetingroom_agenda WHERE meeting_id=@m5_new AND title='리스크 관리 및 문제 해결' ORDER BY id DESC LIMIT 1);
SET @a14_new = (SELECT id FROM meetingroom_agenda WHERE meeting_id=@m5_new AND title='고객 피드백 및 개선 계획' ORDER BY id DESC LIMIT 1);
SET @a15_new = (SELECT id FROM meetingroom_agenda WHERE meeting_id=@m5_new AND title='프로젝트 일정 조정 및 리소스 배분' ORDER BY id DESC LIMIT 1);

--------------------------------------------------------------------------------
-- (c) 각 회의 참여자 등록 (모든 회의에 대해 7명, authority=0)
--------------------------------------------------------------------------------
-- Meeting 1 참여자 등록
INSERT INTO meetingroom_meetingparticipation (authority, meeting_id, participant_id)
VALUES
  (0, @m1_new, (SELECT id FROM accounts_user WHERE name = '김도도')),
  (0, @m1_new, (SELECT id FROM accounts_user WHERE name = '이레레')),
  (0, @m1_new, (SELECT id FROM accounts_user WHERE name = '박미미')),
  (0, @m1_new, (SELECT id FROM accounts_user WHERE name = '정파파')),
  (0, @m1_new, (SELECT id FROM accounts_user WHERE name = '장솔솔')),
  (0, @m1_new, (SELECT id FROM accounts_user WHERE name = '연라라')),
  (0, @m1_new, (SELECT id FROM accounts_user WHERE name = '최시시'));

-- Meeting 2 참여자 등록
INSERT INTO meetingroom_meetingparticipation (authority, meeting_id, participant_id)
VALUES
  (0, @m2_new, (SELECT id FROM accounts_user WHERE name = '김도도')),
  (0, @m2_new, (SELECT id FROM accounts_user WHERE name = '이레레')),
  (0, @m2_new, (SELECT id FROM accounts_user WHERE name = '박미미')),
  (0, @m2_new, (SELECT id FROM accounts_user WHERE name = '정파파')),
  (0, @m2_new, (SELECT id FROM accounts_user WHERE name = '장솔솔')),
  (0, @m2_new, (SELECT id FROM accounts_user WHERE name = '연라라')),
  (0, @m2_new, (SELECT id FROM accounts_user WHERE name = '최시시'));

-- Meeting 3 참여자 등록
INSERT INTO meetingroom_meetingparticipation (authority, meeting_id, participant_id)
VALUES
  (0, @m3_new, (SELECT id FROM accounts_user WHERE name = '김도도')),
  (0, @m3_new, (SELECT id FROM accounts_user WHERE name = '이레레')),
  (0, @m3_new, (SELECT id FROM accounts_user WHERE name = '박미미')),
  (0, @m3_new, (SELECT id FROM accounts_user WHERE name = '정파파')),
  (0, @m3_new, (SELECT id FROM accounts_user WHERE name = '장솔솔')),
  (0, @m3_new, (SELECT id FROM accounts_user WHERE name = '연라라')),
  (0, @m3_new, (SELECT id FROM accounts_user WHERE name = '최시시'));

-- Meeting 4 참여자 등록
INSERT INTO meetingroom_meetingparticipation (authority, meeting_id, participant_id)
VALUES
  (0, @m4_new, (SELECT id FROM accounts_user WHERE name = '김도도')),
  (0, @m4_new, (SELECT id FROM accounts_user WHERE name = '이레레')),
  (0, @m4_new, (SELECT id FROM accounts_user WHERE name = '박미미')),
  (0, @m4_new, (SELECT id FROM accounts_user WHERE name = '정파파')),
  (0, @m4_new, (SELECT id FROM accounts_user WHERE name = '장솔솔')),
  (0, @m4_new, (SELECT id FROM accounts_user WHERE name = '연라라')),
  (0, @m4_new, (SELECT id FROM accounts_user WHERE name = '최시시'));

-- Meeting 5 참여자 등록
INSERT INTO meetingroom_meetingparticipation (authority, meeting_id, participant_id)
VALUES
  (0, @m5_new, (SELECT id FROM accounts_user WHERE name = '김도도')),
  (0, @m5_new, (SELECT id FROM accounts_user WHERE name = '이레레')),
  (0, @m5_new, (SELECT id FROM accounts_user WHERE name = '박미미')),
  (0, @m5_new, (SELECT id FROM accounts_user WHERE name = '정파파')),
  (0, @m5_new, (SELECT id FROM accounts_user WHERE name = '장솔솔')),
  (0, @m5_new, (SELECT id FROM accounts_user WHERE name = '연라라')),
  (0, @m5_new, (SELECT id FROM accounts_user WHERE name = '최시시'));

--------------------------------------------------------------------------------
-- (d) 회의록(Meeting Minutes) 삽입
--------------------------------------------------------------------------------
/* [Meeting 1 – Agenda 1] */
INSERT INTO meetingroom_mom (agenda_result, completed, agenda_id, meeting_id)
VALUES
(
  '자, 첫 번째 안건인 “프로젝트 관련 최신 이슈”부터 시작할까요? 요즘 프로젝트 진행 상황을 보면 몇 가지 이슈가 발생했어요. 가장 큰 문제는 데이터 수집과 처리에서 병목 현상이 발생하고 있다는 점이에요. 우리가 수집한 데이터는 많이 있지만, 이걸 모델 학습에 적합하게 정제하는 데 시간이 너무 많이 걸리고 있죠.

맞아요. 특히 잡음 제거나 텍스트 변환에서 인식 오류가 많이 발생하고 있죠. 실제로 회의 중 배경 소음 때문에 정확도가 떨어지고, 다수의 사람들이 동시에 말할 때 시스템이 헷갈려하는 경우가 많아요. 그래서 이 문제를 어떻게 해결할지 고민하네요.

네, 그래서 지난번부터 실시간 처리 속도를 높이려고 시도는 했지만, 시스템이 너무 과부하되면서 속도가 오히려 느려졌어요. 이 문제를 해결하려면 분산 처리 시스템을 도입해야 할 것 같아요. 이걸 통해 동시에 데이터를 더 빠르게 처리할 수 있을 거예요.

그럼 데이터 처리 속도를 높이기 위한 기술적 접근에 대한 논의는 필요하겠네요. 그리고 정확도를 높이려면 음성 인식 모델을 더 튜닝해야 하는데, 추가 데이터 샘플이 필요한 것 같아요. 최근에 수집한 데이터만으로는 부족할 수 있을 것 같아요.

그 부분에 대해서는 저도 동의해요. 특히 발음이 비슷한 단어들(예: “기술” vs “키츠”)에 대한 인식 정확도를 높이는 게 우선이라고 봅니다.

음, 그럼 우리가 우선적으로 해야 할 일은 데이터셋을 더 확장하고, 음성 인식 성능을 개선하는 것이겠군요. 그리고 회의 환경에서 실시간으로 동작하도록 시스템을 최적화하는 것도 중요한 부분이고요. 이걸 개선하기 위한 방향을 다음 회의 때 구체적으로 정리해보면 좋겠네요.',
  0,
  @a1_new,
  @m1_new
);

-- [Meeting 1 – Agenda 2]
INSERT INTO meetingroom_mom (agenda_result, completed, agenda_id, meeting_id)
VALUES
(
  '두 번째 안건인 “데이터 파이프라인 검토”에 대해서도 이야기해봅시다. 최근 파이프라인을 점검하면서 몇 가지 병목 현상이 발견되었습니다. 특히 텍스트 변환 후 후처리 과정이 너무 길어지고 있어서 최적화가 필요합니다.

맞아요. 데이터를 실시간으로 처리할 시스템이 부족해 병렬 처리나 분산 시스템 도입이 필요할 것 같습니다. 현재 단일 서버 시스템으로는 한계가 있습니다.',
  0,
  @a2_new,
  @m1_new
);

-- [Meeting 1 – Agenda 3]
INSERT INTO meetingroom_mom (agenda_result, completed, agenda_id, meeting_id)
VALUES
(
  '세 번째 안건인 “개발 일정 계획”에 대해 논의합니다. 설정한 일정에 대해 점검한 결과, 몇 가지 일정이 지연되고 있습니다.

특히 데이터 파이프라인 개선과 성능 최적화 작업에 시간이 더 소요되고 있어, 추가 2~3주가 필요할 것으로 보입니다. 회의록 자동 생성 시스템 구현은 약 4주 정도 소요될 것으로 예상되며, 통합 테스트 등 추가 시간이 필요합니다.

각 팀의 업데이트된 계획을 다음 회의에서 공유하면 좋겠습니다.',
  0,
  @a3_new,
  @m1_new
);

-- [Meeting 2 – Agenda 1]
INSERT INTO meetingroom_mom (agenda_result, completed, agenda_id, meeting_id)
VALUES
(
  '첫 번째 안건 “개발 일정 수정”을 논의합니다. 지난 회의에서 설정한 일정대로 진행되고 있는지 점검한 결과, 데이터 파이프라인 구축에서 예상보다 시간이 더 소요되고 있음을 확인했습니다.

텍스트 변환 후 처리 과정에서 발생한 병목 현상으로 인해 일정 수정이 필요해 보입니다. 추가 데이터 수집과 성능 최적화가 요구되며, 약 2주 이상의 추가 기간이 필요할 것으로 예상됩니다.

또한, STT 시스템의 성능 최적화도 연장해야 하며, 원래 2주 계획이 3주 정도로 늘어날 필요가 있습니다.

따라서 6개월 내 출시 목표에 대해 재검토가 필요합니다.',
  0,
  @a4_new,
  @m2_new
);

-- [Meeting 2 – Agenda 2]
INSERT INTO meetingroom_mom (agenda_result, completed, agenda_id, meeting_id)
VALUES
(
  '두 번째 안건 “추천 알고리즘 및 AI 모델 결정”입니다. 회의록 시스템에 적용할 추천 알고리즘에 대해 고민해야 합니다.

추천 시스템은 회의 주제나 아이디어를 기반으로 후속 자료를 추천하는 기능으로, 협업 필터링과 콘텐츠 기반 필터링을 결합하는 방안을 고려할 수 있습니다. 초기에는 콘텐츠 기반 방식으로 시작하고, 사용자가 늘어나면 협업 필터링을 도입하는 방안을 논의합니다.

또한, BERT를 주제 분류에, GPT 계열 모델을 회의록 생성에 활용하는 방안도 검토해볼 만합니다.',
  0,
  @a5_new,
  @m2_new
);

-- [Meeting 2 – Agenda 3]
INSERT INTO meetingroom_mom (agenda_result, completed, agenda_id, meeting_id)
VALUES
(
  '세 번째 안건 “모의 테스트 결과 및 추후 테스트 계획 결정”입니다. 최근 모의 테스트에서 STT 시스템의 정확도와 실시간 처리 속도 문제가 드러났습니다.

각 시스템에 대해 2주 단위로 테스트를 반복하며 개선점을 도출하고, 사용자 피드백을 적극 반영하여 시스템을 보완하는 방안을 마련해야 합니다.',
  0,
  @a6_new,
  @m2_new
);

-- [Meeting 3 – Agenda 1]
INSERT INTO meetingroom_mom (agenda_result, completed, agenda_id, meeting_id)
VALUES
(
  '첫 번째 안건 “제품 마케팅 전략 수립”입니다. 회의록 자동화와 생산성 향상이라는 핵심 가치를 바탕으로, 디지털 및 B2B 마케팅 전략을 세워 초기 사용자를 확보할 방안을 논의합니다.',
  0,
  @a7_new,
  @m3_new
);

-- [Meeting 3 – Agenda 2]
INSERT INTO meetingroom_mom (agenda_result, completed, agenda_id, meeting_id)
VALUES
(
  '두 번째 안건 “브랜드 인지도 향상 방안”입니다. 콘텐츠 마케팅, 인플루언서 협업, 소셜 미디어, 웹 세미나 등을 통해 제품의 가치를 알리고 고객 신뢰도를 높이는 전략을 마련해야 합니다.',
  0,
  @a8_new,
  @m3_new
);

-- [Meeting 3 – Agenda 3]
INSERT INTO meetingroom_mom (agenda_result, completed, agenda_id, meeting_id)
VALUES
(
  '세 번째 안건 “가격 전략 및 경쟁 분석”입니다. 경쟁사 제품과의 가격 비교를 통해, 기본 및 프리미엄 요금제 설정, 구독 할인 및 사용량 기반 요금 모델 도입 방안을 논의합니다.',
  0,
  @a9_new,
  @m3_new
);

-- [Meeting 4 – Agenda 1]
INSERT INTO meetingroom_mom (agenda_result, completed, agenda_id, meeting_id)
VALUES
(
  '첫 번째 안건 “수익화 모델 정의”입니다. 구독, 사용량 기반, 프리미엄(freemium) 모델을 복합적으로 도입하여 소규모부터 대기업까지 다양한 고객층을 커버할 수 있는 수익화 전략을 마련하는 방안을 검토합니다.',
  0,
  @a10_new,
  @m4_new
);

-- [Meeting 4 – Agenda 2]
INSERT INTO meetingroom_mom (agenda_result, completed, agenda_id, meeting_id)
VALUES
(
  '두 번째 안건 “가격 전략 및 경쟁 분석”입니다. 경쟁사 제품의 가격대를 참고하여 기본 및 프리미엄 요금제와 기업 맞춤형 가격 정책을 마련하는 방안을 논의합니다.',
  0,
  @a11_new,
  @m4_new
);

-- [Meeting 5 – Agenda 1]
INSERT INTO meetingroom_mom (agenda_result, completed, agenda_id, meeting_id)
VALUES
(
  '첫 번째 안건 “프로젝트 진행 상황 점검”입니다. 데이터 수집 및 음성 인식 시스템의 성능 개선 작업이 예상보다 지연되고 있음을 점검하고, 추가 데이터 수집 및 분산 처리 시스템 도입 방안을 논의합니다.',
  0,
  @a12_new,
  @m5_new
);

-- [Meeting 5 – Agenda 2]
INSERT INTO meetingroom_mom (agenda_result, completed, agenda_id, meeting_id)
VALUES
(
  '두 번째 안건 “리스크 관리 및 문제 해결”입니다. 음성 인식 오류와 실시간 처리 지연 등의 리스크를 파악하고, 분산 처리 도입 및 모델 튜닝 등 구체적인 해결 방안을 논의합니다.',
  0,
  @a13_new,
  @m5_new
);

-- [Meeting 5 – Agenda 3]
INSERT INTO meetingroom_mom (agenda_result, completed, agenda_id, meeting_id)
VALUES
(
  '세 번째 안건 “고객 피드백 및 개선 계획”입니다. 베타 테스트를 통해 받은 고객 피드백—회의록 정확도, 처리 속도 개선 요청 및 추천 시스템 개선 요구—를 바탕으로 개선 방향을 논의합니다.',
  0,
  @a14_new,
  @m5_new
);

-- [Meeting 5 – Agenda 4]
INSERT INTO meetingroom_mom (agenda_result, completed, agenda_id, meeting_id)
VALUES
(
  '네 번째 안건 “프로젝트 일정 조정 및 리소스 배분”입니다. 음성 인식 및 데이터 처리 개선 작업으로 인한 일정 지연을 반영하여, 전체 프로젝트 일정을 재조정하고 각 팀에 적절한 리소스를 배분하는 방안을 논의합니다.',
  0,
  @a15_new,
  @m5_new
);

--------------------------------------------------------------------------------
-- (e) 안건별 문서 생성 및 meetingroom_mom 업데이트
--------------------------------------------------------------------------------
-- Meeting 1 문서 생성 및 업데이트
INSERT INTO projects_document (type, embedding, department_id, project_id)
VALUES (2, 1, (SELECT id FROM accounts_department WHERE name = '개발'), @project_id);
SET @doc1 = LAST_INSERT_ID();
UPDATE meetingroom_mom SET document_id = @doc1 WHERE agenda_id = @a1_new;

INSERT INTO projects_document (type, embedding, department_id, project_id)
VALUES (2, 1, (SELECT id FROM accounts_department WHERE name = '개발'), @project_id);
SET @doc2 = LAST_INSERT_ID();
UPDATE meetingroom_mom SET document_id = @doc2 WHERE agenda_id = @a2_new;

INSERT INTO projects_document (type, embedding, department_id, project_id)
VALUES (2, 1, (SELECT id FROM accounts_department WHERE name = '개발'), @project_id);
SET @doc3 = LAST_INSERT_ID();
UPDATE meetingroom_mom SET document_id = @doc3 WHERE agenda_id = @a3_new;

-- Meeting 2 문서 생성 및 업데이트
INSERT INTO projects_document (type, embedding, department_id, project_id)
VALUES (2, 1, (SELECT id FROM accounts_department WHERE name = '개발'), @project_id);
SET @doc4 = LAST_INSERT_ID();
UPDATE meetingroom_mom SET document_id = @doc4 WHERE agenda_id = @a4_new;

INSERT INTO projects_document (type, embedding, department_id, project_id)
VALUES (2, 1, (SELECT id FROM accounts_department WHERE name = '개발'), @project_id);
SET @doc5 = LAST_INSERT_ID();
UPDATE meetingroom_mom SET document_id = @doc5 WHERE agenda_id = @a5_new;

INSERT INTO projects_document (type, embedding, department_id, project_id)
VALUES (2, 1, (SELECT id FROM accounts_department WHERE name = '개발'), @project_id);
SET @doc6 = LAST_INSERT_ID();
UPDATE meetingroom_mom SET document_id = @doc6 WHERE agenda_id = @a6_new;

-- Meeting 3 문서 생성 및 업데이트
INSERT INTO projects_document (type, embedding, department_id, project_id)
VALUES (2, 1, (SELECT id FROM accounts_department WHERE name = '개발'), @project_id);
SET @doc7 = LAST_INSERT_ID();
UPDATE meetingroom_mom SET document_id = @doc7 WHERE agenda_id = @a7_new;

INSERT INTO projects_document (type, embedding, department_id, project_id)
VALUES (2, 1, (SELECT id FROM accounts_department WHERE name = '개발'), @project_id);
SET @doc8 = LAST_INSERT_ID();
UPDATE meetingroom_mom SET document_id = @doc8 WHERE agenda_id = @a8_new;

INSERT INTO projects_document (type, embedding, department_id, project_id)
VALUES (2, 1, (SELECT id FROM accounts_department WHERE name = '개발'), @project_id);
SET @doc9 = LAST_INSERT_ID();
UPDATE meetingroom_mom SET document_id = @doc9 WHERE agenda_id = @a9_new;

-- Meeting 4 문서 생성 및 업데이트
INSERT INTO projects_document (type, embedding, department_id, project_id)
VALUES (2, 1, (SELECT id FROM accounts_department WHERE name = '개발'), @project_id);
SET @doc10 = LAST_INSERT_ID();
UPDATE meetingroom_mom SET document_id = @doc10 WHERE agenda_id = @a10_new;

INSERT INTO projects_document (type, embedding, department_id, project_id)
VALUES (2, 1, (SELECT id FROM accounts_department WHERE name = '개발'), @project_id);
SET @doc11 = LAST_INSERT_ID();
UPDATE meetingroom_mom SET document_id = @doc11 WHERE agenda_id = @a11_new;

-- Meeting 5 문서 생성 및 업데이트
INSERT INTO projects_document (type, embedding, department_id, project_id)
VALUES (2, 1, (SELECT id FROM accounts_department WHERE name = '개발'), @project_id);
SET @doc12 = LAST_INSERT_ID();
UPDATE meetingroom_mom SET document_id = @doc12 WHERE agenda_id = @a12_new;

INSERT INTO projects_document (type, embedding, department_id, project_id)
VALUES (2, 1, (SELECT id FROM accounts_department WHERE name = '개발'), @project_id);
SET @doc13 = LAST_INSERT_ID();
UPDATE meetingroom_mom SET document_id = @doc13 WHERE agenda_id = @a13_new;

INSERT INTO projects_document (type, embedding, department_id, project_id)
VALUES (2, 1, (SELECT id FROM accounts_department WHERE name = '개발'), @project_id);
SET @doc14 = LAST_INSERT_ID();
UPDATE meetingroom_mom SET document_id = @doc14 WHERE agenda_id = @a14_new;

INSERT INTO projects_document (type, embedding, department_id, project_id)
VALUES (2, 1, (SELECT id FROM accounts_department WHERE name = '개발'), @project_id);
SET @doc15 = LAST_INSERT_ID();
UPDATE meetingroom_mom SET document_id = @doc15 WHERE agenda_id = @a15_new;